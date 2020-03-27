{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
{-# LANGUAGE DeriveAnyClass      #-}
{-# LANGUAGE DeriveGeneric       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections       #-}
{-# LANGUAGE ViewPatterns        #-}
module ML.BORL.Step
    ( step
    , steps
    , stepM
    , stepsM
    , restoreTensorflowModels
    , saveTensorflowModels
    , stepExecute
    , nextAction
    , epsCompareWith
    , sortBy
    ) where

#ifdef DEBUG
import           Control.Concurrent.MVar
import           System.IO.Unsafe                   (unsafePerformIO)
#endif
import           Control.Applicative                ((<|>))
import           Control.Arrow                      ((&&&), (***))
import           Control.DeepSeq
import           Control.DeepSeq                    (NFData, force)
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class             (liftIO)
import           Control.Monad.IO.Class             (MonadIO, liftIO)
import           Control.Parallel.Strategies        hiding (r0)
import           Data.Function                      (on)
import           Data.List                          (find, groupBy, intercalate, partition,
                                                     sortBy, transpose)
import qualified Data.Map.Strict                    as M
import           Data.Maybe                         (fromMaybe, isJust)
import           Data.Serialize
import           GHC.Generics
import           System.Directory
import           System.IO
import           System.Random
import           Text.Printf


import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Calculation
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork.NNConfig
import           ML.BORL.NeuralNetwork.ReplayMemory
import           ML.BORL.NeuralNetwork.Scaling
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                      as P
import           ML.BORL.Reward
import           ML.BORL.SaveRestore
import           ML.BORL.Serialisable
import           ML.BORL.Type
import           ML.BORL.Types
import           ML.BORL.Workers.Type
import           ML.BORL.Workers.Type


import           Debug.Trace

fileDebugStateV :: FilePath
fileDebugStateV = "stateVAllStates"

fileDebugStateW :: FilePath
fileDebugStateW = "stateWAllStates"

fileDebugPsiVValues :: FilePath
fileDebugPsiVValues = "statePsiVAllStates"

fileDebugPsiWValues :: FilePath
fileDebugPsiWValues = "statePsiWAllStates"

fileStateValues :: FilePath
fileStateValues = "stateValues"

fileDebugStateValuesNrStates :: FilePath
fileDebugStateValuesNrStates = "stateValuesAllStatesCount"

fileReward :: FilePath
fileReward = "reward"

fileEpisodeLength :: FilePath
fileEpisodeLength = "episodeLength"


steps :: (NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> IO (BORL s)
steps (force -> borl) nr =
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> runMonadBorlIO $ force <$> foldM (\b _ -> nextAction (force b) >>= stepExecute b) borl [0 .. nr - 1]
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- foldM (\b _ -> nextAction (force b) >>= stepExecute b) borl [0 .. nr - 1]
        force <$> saveTensorflowModels borl'


step :: (NFData s, Ord s, RewardFuture s) => BORL s -> IO (BORL s)
step (force -> borl) =
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> nextAction borl >>= stepExecute borl
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- nextAction borl >>= stepExecute borl
        force <$> saveTensorflowModels borl'

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to step.
stepM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> m (BORL s)
stepM (force -> borl) = nextAction (force borl) >>= stepExecute borl

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to steps, but forces
-- evaluation of the data structure every 1000 periods.
stepsM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> m (BORL s)
stepsM (force -> borl) nr = do
  !borl' <- force <$> foldM (\b _ -> nextAction (force b) >>= stepExecute b) borl [1 .. min maxNr nr]
  if nr > maxNr
    then stepsM borl' (nr - maxNr)
    else return borl'
  where maxNr = 1000

stepExecute :: forall m s . (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> NextActions s -> m (BORL s)
stepExecute borl ((randomAction, (aNr, Action action _)), workerActions)
  -- File IO Operations
 = do
  let state = borl ^. s
      period = borl ^. t + length (borl ^. futureRewards)
  when (period == 0) $ do
    liftIO $ writeFile fileStateValues "Period\tRho\tMinRho\tVAvg\tR0\tR1\n"
    liftIO $ writeFile fileEpisodeLength "Episode\tEpisodeLength\n"
    liftIO $ writeFile fileReward "Period\tReward\n"
  workerReplMemFuture <- liftIO $ doFork $ runWorkerActions borl workerActions
  (reward, stateNext, episodeEnd) <- liftIO $ action state
  let applyToReward (RewardFuture storage storageWorkers) = applyState storage state
      applyToReward r                                     = r
      updateFutures = map (over futureReward applyToReward)
  let borl' = over futureRewards (updateFutures . (++ [RewardFutureData period state aNr randomAction reward stateNext episodeEnd])) borl
  (dropLen, _, borlNew) <- foldM stepExecuteMaterialisedFutures (0, False, borl') (borl' ^. futureRewards)
  (workerReplMems', workerFutureRewards') <- liftIO $ collectForkResult workerReplMemFuture -- Note that the replay memory of the workers are offset by 1 step
  return $
    force $
    set (workers . traversed . workersReplayMemories) workerReplMems' $
    set (workers . traversed . workersFutureRewards) workerFutureRewards' $ over futureRewards (drop dropLen) $ set s stateNext borlNew

-- | This functions takes one step for all workers, and returns the new worker replay memories and future reward data
-- lists.
runWorkerActions :: BORL s -> [WorkerActionChoice s] -> IO ([ReplayMemories], [[RewardFutureData s]])
runWorkerActions borl acts = do
  let states = borl ^. workers.traversed.workersS
  stepRewards <- zipWithM runWorkerAction states acts
  let futureRews = borl ^. workers.traversed.workersFutureRewards
  let updateFuturesWith f = map (over futureReward f)
  let futureRewsTmp = zipWith3 (\state fs r -> updateFuturesWith (applyToReward state) (fs ++ [r])) states futureRews stepRewards
  let (rewards, futureRews') = unzip $ map splitMaterialisedFutures futureRewsTmp
  (,futureRews') <$> zipWithM (foldM addExperience) (borl ^. workers.traversed.workersReplayMemories) rewards
  where runWorkerAction :: (MonadBorl' m) => State s -> WorkerActionChoice s -> m (RewardFutureData s)
        runWorkerAction state (randomAction, (aNr, Action action _)) = do
          (reward, stateNext, episodeEnd) <- liftIO $ action state
          return $ RewardFutureData (borl ^. t) state aNr randomAction reward stateNext episodeEnd
        applyToReward state (RewardFuture _ storageWorkers) = applyState storageWorkers state
        applyToReward _ r                               = r
        splitMaterialisedFutures fs = let xs = takeWhile (not . isRewardFuture . view futureReward) fs
          in (filter (not . isRewardEmpty . view futureReward) xs, drop (length xs) fs)
        addExperience replMem (RewardFutureData _ state aNr randomAction (Reward reward) stateNext episodeEnd) = do
          let (_, stateActs, stateNextActs) = mkStateActs borl state stateNext
          liftIO $ addToReplayMemories (stateActs, aNr, randomAction, reward, stateNextActs, episodeEnd) replMem
        addExperience _ _ = error "Unexpected Reward in calcExperience of runWorkerActions! "

-- | This function exectues all materialised rewards until a non-materialised reward is found, i.e. add a new experience
-- to the replay memory and then, select and learn from the experiences of the replay memory.
stepExecuteMaterialisedFutures ::
     forall m s. (MonadBorl' m, NFData s, Ord s, RewardFuture s)
  => (Int, Bool, BORL s)
  -> RewardFutureData s
  -> m (Int, Bool, BORL s)
stepExecuteMaterialisedFutures (nr, True, borl) _ = return (nr, True, borl)
stepExecuteMaterialisedFutures (nr, _, borl) dt =
  case view futureReward dt of
    RewardEmpty     -> return (nr, False, borl)
    RewardFuture {} -> return (nr, True, borl)
    Reward {}       -> (nr+1, False, ) <$> execute borl dt

-- | Execute the given step, i.e. add a new experience to the replay memory and then, select and learn from the
-- experiences of the replay memory.
execute :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> RewardFutureData s -> m (BORL s)
execute borl (RewardFutureData period state aNr randomAction (Reward reward) stateNext episodeEnd) = do
#ifdef DEBUG
  when (borl ^. t == 0) $ forM_ [fileDebugStateV, fileDebugStateW, fileDebugPsiWValues, fileDebugPsiVValues, fileDebugPsiWValues, fileDebugStateValuesNrStates] $ \f ->
    liftIO $ doesFileExist f >>= \x -> when x (removeFile f)
  borl <- writeDebugFiles borl
#endif
  (proxies', calc) <- P.insert borl period state aNr randomAction reward stateNext episodeEnd (mkCalculation borl) (borl ^. proxies)
  let lastVsLst = fromMaybe [0] (getLastVs' calc)
  let strRho = show (fromMaybe 0 (getRhoVal' calc))
      strMinV = show (fromMaybe 0 (getRhoMinimumVal' calc))
      strVAvg = show (avg lastVsLst)
      strR0 = show $ fromMaybe 0 (getR0ValState' calc)
      strR1 = show $ fromMaybe 0 (getR1ValState' calc)
      avg xs = sum xs / fromIntegral (length xs)
  liftIO $ appendFile fileStateValues (show period ++ "\t" ++ strRho ++ "\t" ++ strMinV ++ "\t" ++ strVAvg ++ "\t" ++ strR0 ++ "\t" ++ strR1 ++ "\n")
  let (eNr, eStart) = borl ^. episodeNrStart
      eLength = borl ^. t - eStart
  when (getEpisodeEnd calc) $ liftIO $ appendFile fileEpisodeLength (show eNr ++ "\t" ++ show eLength ++ "\n")
  liftIO $ appendFile fileReward (show period ++ "\t" ++ show reward ++ "\n")
  -- update values
  let setEpisode curEp
        | getEpisodeEnd calc = (eNr + 1, borl ^. t)
        | otherwise = curEp
  return $
    set psis (fromMaybe 0 (getPsiValRho' calc), fromMaybe 0 (getPsiValV' calc), fromMaybe 0 (getPsiValW' calc)) $
    set lastVValues (fromMaybe [] (getLastVs' calc)) $ set lastRewards (getLastRews' calc) $ set proxies proxies' $ set t (period + 1) $ over episodeNrStart setEpisode borl
execute _ _ = error "Exectue on invalid data structure. This is a bug!"


#ifdef DEBUG

stateFeatures :: MVar [a]
stateFeatures = unsafePerformIO $ newMVar mempty
{-# NOINLINE stateFeatures #-}

setStateFeatures :: (MonadIO m) => [a] -> m ()
setStateFeatures x = liftIO $ modifyMVar_ stateFeatures (return . const x)

getStateFeatures :: (MonadIO m) => m [a]
getStateFeatures = liftIO $ fromMaybe mempty <$> tryReadMVar stateFeatures


writeDebugFiles :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> m (BORL s)
writeDebugFiles borl = do
  let isDqn = isAlgDqn (borl ^. algorithm) || isAlgDqnAvgRewardFree (borl ^. algorithm)
  let isAnn
        | isDqn = P.isNeuralNetwork (borl ^. proxies . r1)
        | otherwise = P.isNeuralNetwork (borl ^. proxies . v)
  let putStateFeatList borl xs
        | isAnn = borl
        | otherwise = setAllProxies proxyTable xs' borl
        where
          xs' = M.fromList $ zip (map (\xs -> (init xs, round (last xs))) xs) (repeat 0)
  borl' <-
    if borl ^. t > 0
      then return borl
      else do
        liftIO $ writeFile fileDebugStateV ""
        liftIO $ writeFile fileDebugStateW ""
        liftIO $ writeFile fileDebugPsiVValues ""
        liftIO $ writeFile fileDebugPsiWValues ""
        borl' <-
          if isAnn
            then return borl
            else stepsM
                   (setAllProxies (proxyNNConfig . trainMSEMax) Nothing $ setAllProxies (proxyNNConfig . replayMemoryMaxSize) 1000 $ set t 1 borl)
                   debugStepsCount -- run steps to fill the table with (hopefully) all states
        let stateFeats
              | isDqn = getStateFeatList (borl' ^. proxies . r1)
              | otherwise = getStateFeatList (borl' ^. proxies . v)
        setStateFeatures stateFeats
        liftIO $ writeFile fileDebugStateValuesNrStates (show $ length stateFeats)
        liftIO $ forM_ [fileDebugStateV, fileDebugStateW, fileDebugPsiVValues, fileDebugPsiWValues] $ flip writeFile ("Period\t" <> mkListStr (shorten . printFloat) stateFeats <> "\n")
        if isNeuralNetwork (borl ^. proxies . v)
          then return borl
          else do
            liftIO $ putStrLn $ "[DEBUG INFERRED NUMBER OF STATES]: " <> show (length stateFeats)
            return $ putStateFeatList borl stateFeats
  let stateFeats
        | isDqn = getStateFeatList (borl' ^. proxies . r1)
        | otherwise = getStateFeatList (borl' ^. proxies . v)
      isTf
        | isDqn && isTensorflow (borl' ^. proxies . r1) = True
        | isTensorflow (borl' ^. proxies . v) = True
        | otherwise = False
  stateFeats <- getStateFeatures
  when ((borl' ^. t `mod` debugPrintCount) == 0) $ do
    stateValuesV <- mapM (\xs -> if isDqn then rValueWith Worker borl' RBig (init xs) (round $ last xs) else vValueWith Worker borl' (init xs) (round $ last xs)) stateFeats
    stateValuesW <- mapM (\xs -> if isDqn then return 0 else wValueFeat borl' (init xs) (round $ last xs)) stateFeats
    liftIO $ appendFile fileDebugStateV (show (borl' ^. t) <> "\t" <> mkListStr show stateValuesV <> "\n")
    when (isAlgBorl (borl ^. algorithm)) $ do
      liftIO $ appendFile fileDebugStateW (show (borl' ^. t) <> "\t" <> mkListStr show stateValuesW <> "\n")
      psiVValues <- mapM (\xs -> psiVFeat borl' (init xs) (round $ last xs)) stateFeats
      liftIO $ appendFile fileDebugPsiVValues (show (borl' ^. t) <> "\t" <> mkListStr show psiVValues <> "\n")
      psiWValues <- mapM (\xs -> psiWFeat borl' (init xs) (round $ last xs)) stateFeats
      liftIO $ appendFile fileDebugPsiWValues (show (borl' ^. t) <> "\t" <> mkListStr show psiWValues <> "\n")
  return borl'
  where
    getStateFeatList Scalar {} = []
    getStateFeatList (Table t _) = map (\(xs, y) -> xs ++ [fromIntegral y]) (M.keys t)
    getStateFeatList nn = concatMap (\xs -> map (\(idx, _) -> xs ++ [fromIntegral idx]) acts) (nn ^. proxyNNConfig . prettyPrintElems)
    acts = borl ^. actionList
    mkListStr :: (a -> String) -> [a] -> String
    mkListStr f = intercalate "\t" . map f
    shorten xs | length xs > 60 = "..." <> drop (length xs - 60) xs
               | otherwise = xs
    printFloat :: [Double] -> String
    printFloat xs = "[" <> intercalate "," (map (printf "%.2f") xs) <> "]"
    psiVFeat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiV)
    psiWFeat borl stateFeat aNr = P.lookupProxy (borl ^. t) Worker (stateFeat, aNr) (borl ^. proxies . psiW)

debugStepsCount :: Integer
debugStepsCount = 8000

debugPrintCount :: Int
debugPrintCount = 100

#endif
