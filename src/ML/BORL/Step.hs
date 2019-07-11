{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE CPP                 #-}
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

import           ML.BORL.Action
import           ML.BORL.Algorithm
import           ML.BORL.Calculation
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork.Tensorflow (buildTensorflowModel,
                                                   restoreModelWithLastIO,
                                                   saveModelWithLastIO)
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Proxy                    as P
import           ML.BORL.Reward
import           ML.BORL.SaveRestore
import           ML.BORL.Serialisable
import           ML.BORL.Type
import           ML.BORL.Types

import           Control.Applicative              ((<|>))
import           Control.DeepSeq                  (NFData, force)
import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class           (MonadIO, liftIO)
import           Control.Parallel.Strategies      hiding (r0)
import           Data.Function                    (on)
import           Data.List                        (find, groupBy, sortBy)
import qualified Data.Map.Strict                  as M
import           Data.Maybe                       (fromMaybe, isJust)
import           System.Directory
import           System.IO
import           System.Random

import           Debug.Trace


fileStateValues :: FilePath
fileStateValues = "stateValues"

fileEpisodeLength :: FilePath
fileEpisodeLength = "episodeLength"


steps :: (NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> IO (BORL s)
steps (force -> borl) nr =
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> runMonadBorlIO $ force <$> foldM (\b _ -> nextAction (force b) >>= fmap force . stepExecute) borl [0 .. nr - 1]
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- foldM (\b _ -> nextAction (force b) >>= fmap force . stepExecute) borl [0 .. nr - 1]
        force <$> saveTensorflowModels borl'


step :: (NFData s, Ord s, RewardFuture s) => BORL s -> IO (BORL s)
step (force -> borl) =
  case find isTensorflow (allProxies $ borl ^. proxies) of
    Nothing -> nextAction borl >>= stepExecute
    Just _ ->
      runMonadBorlTF $ do
        void $ restoreTensorflowModels True borl
        !borl' <- nextAction borl >>= stepExecute
        force <$> saveTensorflowModels borl'

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to step.
stepM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> m (BORL s)
stepM (force -> borl) = nextAction borl >>= fmap force . stepExecute

-- | This keeps the Tensorflow session alive. For non-Tensorflow BORL data structures this is equal to steps, but forces
-- evaluation of the data structure every 1000 periods.
stepsM :: (MonadBorl' m, NFData s, Ord s, RewardFuture s) => BORL s -> Integer -> m (BORL s)
stepsM (force -> borl) nr = do
  !borl' <- force <$> foldM (\b _ -> nextAction b >>= stepExecute) borl [1 .. min maxNr nr]
  if nr > maxNr
    then stepsM borl' (nr - maxNr)
    else return borl'
  where maxNr = 1000


-- | This function chooses the next action from the current state s and all possible actions.
nextAction :: (MonadBorl' m, Ord s) => BORL s -> m (BORL s, Bool, ActionIndexed s)
nextAction borl
  | null as = error "Empty action list"
  | length as == 1 = return (borl, False, head as)
  | otherwise = do
    rand <- liftSimple $ randomRIO (0, 1)
    if rand < explore
      then do
        r <- liftSimple $ randomRIO (0, length as - 1)
        return (borl, True, as !! r)
      else case borl ^. algorithm of
             AlgBORL _ _ _ _ decideVPlusPsi -> do
               bestRho <-
                 if isUnichain borl
                   then return as
                   else do
                     rhoVals <- mapM (rhoValue borl state . fst) as
                     return $ map snd $ headRho $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip rhoVals as)
               bestV <-
                 do vVals <- mapM (vValue decideVPlusPsi borl state . fst) bestRho
                    return $ map snd $ headV $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (zip vVals bestRho)
               bestE <-
                 do eVals <- mapM (eValue borl state . fst) bestV
                    return $ map snd $ sortBy (epsCompare compare `on` fst) (zip eVals bestV)
               if length bestE > 1
                 then do
                   r <- liftSimple $ randomRIO (0, length bestE - 1)
                   return (borl, False, bestE !! r)
                 else return (borl, False, headE bestE)
             AlgDQN {} -> dqnNextAction
             AlgDQNAvgRew {} -> dqnNextAction
  where
    headRho []    = error "head: empty input data in nextAction on Rho value"
    headRho (x:_) = x
    headV []    = error "head: empty input data in nextAction on V value"
    headV (x:_) = x
    headE []    = error "head: empty input data in nextAction on E Value"
    headE (x:_) = x
    headDqn []    = error "head: empty input data in nextAction on Dqn Value"
    headDqn (x:_) = x
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
    eps = params' ^. epsilon
    explore = params' ^. exploration
    state = borl ^. s
    as = actionsIndexed borl state
    epsCompare = epsCompareWith eps
    dqnNextAction = do
      rValues <- mapM (rValue borl RBig state . fst) as
      let bestR = sortBy (epsCompare compare `on` fst) (zip rValues as)
               -- liftSimple $ putStrLn ("bestR: " ++ show bestR)
      return (borl, False, snd $ headDqn bestR)

epsCompareWith :: (Ord t, Num t) => t -> (t -> t -> p) -> t -> t -> p
epsCompareWith eps f x y
  | abs (x - y) <= eps = f 0 0
  | otherwise = y `f` x


stepExecute :: forall m s . (MonadBorl' m, NFData s, Ord s, RewardFuture s) => (BORL s, Bool, ActionIndexed s) -> m (BORL s)
stepExecute (borl, randomAction, (aNr, Action action _)) = do
  let state = borl ^. s
      period = borl ^. t
  (reward, stateNext, episodeEnd) <- liftSimple $ action state
  let applyToReward r@(RewardFuture storage) = applyState storage state
      applyToReward r                        = r
      updateFutures = map (over futureReward applyToReward)
  let borl' = over futureRewards (updateFutures . (++ [RewardFutureData period state aNr randomAction reward stateNext episodeEnd])) borl
  borlNew <- snd <$> foldM stepExecuteMaterialisedFutures (False, borl') (borl' ^. futureRewards)
  let dropLen = fromIntegral $ borlNew ^. t - period
  return $ force $ over futureRewards (drop dropLen) $ set s stateNext borlNew


stepExecuteMaterialisedFutures :: forall m s . (MonadBorl' m, NFData s, Ord s) => (Bool, BORL s) -> RewardFutureData s -> m (Bool, BORL s)
stepExecuteMaterialisedFutures (True, borl) _ = return (True, borl)
stepExecuteMaterialisedFutures (_, borl) dt =
  case view futureReward dt of
    RewardEmpty     -> return (False, borl)
    RewardFuture {} -> return (True, borl)
    Reward {}       -> (False, ) <$> execute borl dt

execute :: (MonadBorl' m, NFData s, Ord s) => BORL s -> RewardFutureData s -> m (BORL s)
execute borl (RewardFutureData period state aNr randomAction (Reward reward) stateNext episodeEnd) = do
  (proxies', calc) <- P.insert borl period state aNr randomAction reward stateNext episodeEnd (mkCalculation borl) (borl ^. proxies)
  let lastVsLst = fromMaybe [0] (getLastVs' calc)
  -- File IO Operations
  when (period == 0) $ do
    liftSimple $ writeFile fileStateValues "Period\tRho\tMinRho\tVAvg\tR0\tR1\n"
    liftSimple $ writeFile fileEpisodeLength "Episode\tEpisodeLength\n"
  let strRho = show (fromMaybe 0 (getRhoVal' calc))
      strMinV = show (fromMaybe 0 (getRhoMinimumVal' calc))
      strVAvg = show (avg lastVsLst)
      strR0 = show $ fromMaybe 0 (getR0ValState' calc)
      strR1 = show $ getR1ValState' calc
      divideAfterGrowth :: BORL s -> BORL s
      divideAfterGrowth borl =
        case borl ^. algorithm of
          AlgBORL _ _ _ (DivideValuesAfterGrowth nr maxPeriod) _
            | period > maxPeriod -> borl
            | length lastVsLst == nr && endOfIncreasedStateValues ->
              trace ("multiply in period " ++ show period ++ " by " ++ show val) $
              foldl (\q f -> over (proxies . f) (multiplyProxy val) q) (set phase SteadyStateValues borl) [psiV, v, w]
            where endOfIncreasedStateValues =
                    borl ^. phase == IncreasingStateValues && 2000 * (avg (take (nr `div` 2) lastVsLst) - avg (drop (nr `div` 2) lastVsLst)) / fromIntegral nr < 0.00
                  val = 0.2 / (sum lastVsLst / fromIntegral nr)
          _ -> borl
      setCurrentPhase :: BORL s -> BORL s
      setCurrentPhase borl =
        case borl ^. algorithm of
          AlgBORL _ _ _ (DivideValuesAfterGrowth nr _) _
            | length lastVsLst == nr && increasingStateValue -> trace ("period: " ++ show period) $ trace (show IncreasingStateValues) $ set phase IncreasingStateValues borl
            where increasingStateValue =
                    borl ^. phase /= IncreasingStateValues && 2000 * (avg (take (nr `div` 2) lastVsLst) - avg (drop (nr `div` 2) lastVsLst)) / fromIntegral nr > 0.20
          _ -> borl
      avg xs = sum xs / fromIntegral (length xs)
  liftSimple $ appendFile fileStateValues (show period ++ "\t" ++ strRho ++ "\t" ++ strMinV ++ "\t" ++ strVAvg ++ "\t" ++ strR0 ++ "\t" ++ strR1 ++ "\n")
  let (eNr, eStart) = borl ^. episodeNrStart
      eLength = borl ^. t - eStart
  when (getEpisodeEnd calc) $ liftSimple $ appendFile fileEpisodeLength (show eNr ++ "\t" ++ show eLength ++ "\n")
  -- update values
  let setEpisode curEp
        | getEpisodeEnd calc = (eNr + 1, borl ^. t)
        | otherwise = curEp
  return $
    setCurrentPhase $
    divideAfterGrowth $
    set psis (fromMaybe 0 (getPsiValRho' calc), fromMaybe 0 (getPsiValV' calc), fromMaybe 0 (getPsiValW' calc)) $
    set lastVValues (fromMaybe [] (getLastVs' calc)) $ set lastRewards (getLastRews' calc) $ set proxies proxies' $ set t (period + 1) $ over episodeNrStart setEpisode borl
