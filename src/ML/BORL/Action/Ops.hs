module ML.BORL.Action.Ops
    ( nextAction
    , epsCompareWith
    ) where


import           ML.BORL.Algorithm
import           ML.BORL.Calculation
import           ML.BORL.Exploration
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Type
import           ML.BORL.Types


import           Control.Arrow          ((***))
import           Control.Lens
import           Control.Monad.IO.Class (liftIO)
import           Data.Function          (on)
import           Data.List              (groupBy, partition, sortBy)
import           System.Random

import           Debug.Trace


-- | This function chooses the next action from the current state s and all possible actions.
nextAction :: (MonadBorl' m) => BORL s -> m (BORL s, Bool, ActionIndexed s)
nextAction borl
  | null as = error "Empty action list"
  | length as == 1 = return (borl, False, head as)
  | otherwise =
    case borl ^. parameters . explorationStrategy of
      EpsilonGreedy -> chooseAction borl True (\xs -> return $ SelectedActions (head xs) (last xs))
      SoftmaxBoltzmann t0 -> chooseAction borl False (chooseBySoftmax (t0 * params' ^. exploration))
  where
    as = actionsIndexed borl state
    state = borl ^. s
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)

data SelectedActions s = SelectedActions
  { maximised :: [(Double, ActionIndexed s)] -- ^ Choose actions by maximising
  , minimised :: [(Double, ActionIndexed s)] -- ^ Choose actions by minimising
  }


type UseRand = Bool
type ActionSelection s = [[(Double, ActionIndexed s)]] -> IO (SelectedActions s) -- ^ Incoming actions are sorted with highest value in the head.
type RandomNormValue = Double

chooseBySoftmax :: TemperatureInitFactor -> ActionSelection s
chooseBySoftmax temp xs = do
  r <- liftIO $ randomRIO (0 :: Double, 1)
  return $ SelectedActions (xs !! chooseByProbability r 0 0 probs) (reverse xs !! chooseByProbability r 0 0 probs)
  where
    probs = softmax temp $ map (fst . head) xs

chooseByProbability :: RandomNormValue -> ActionIndex -> Double -> [Double] -> Int
chooseByProbability r idx acc [] = error $ "no more options in chooseByProbability in Action.Ops: " ++ show (r, idx, acc)
chooseByProbability r idx acc (v:vs)
  | acc + v >= r = idx
  | otherwise = chooseByProbability r (idx + 1) (acc + v) vs

chooseAction :: (MonadBorl' m) => BORL s -> UseRand -> ActionSelection s -> m (BORL s, Bool, ActionIndexed s)
chooseAction borl useRand selFromList = do
  rand <- liftIO $ randomRIO (0, 1)
  if useRand && rand < explore
    then do
      r <- liftIO $ randomRIO (0, length as - 1)
      return (borl, True, as !! r)
    else case borl ^. algorithm of
           AlgBORL ga0 ga1 _ _ -> do
             bestRho <-
               if isUnichain borl
                 then return as
                 else do
                   rhoVals <- mapM (rhoValue borl state . fst) as
                   map snd . maximised <$> liftIO (selFromList $ groupBy (epsCompare (==) `on` fst) $ sortBy (compare `on` fst) (zip rhoVals as))
             bestV <-
               do vVals <- mapM (vValue borl state . fst) bestRho
                  map snd . maximised <$> liftIO (selFromList $ groupBy (epsCompare (==) `on` fst) $ sortBy (compare `on` fst) (zip vVals bestRho))
             if length bestV == 1
               then return (borl, False, head bestV)
               else do
                 bestE <-
                   do eVals <- mapM (eValueAvgCleaned borl state . fst) bestV
                      let (increasing, decreasing) = partition ((0 >) . fst) (zip eVals bestV)
                          actionsToChooseFrom
                            | null decreasing = increasing
                            | otherwise = decreasing
                      map snd . maximised <$> liftIO (selFromList $ groupBy (epsCompareWith (ga1 - ga0) (==) `on` fst) $ sortBy (compare `on` fst) actionsToChooseFrom)
                 -- other way of doing it:
                 -- ----------------------
                 -- do eVals <- mapM (eValue borl state . fst) bestV
                 --    rhoVal <- rhoValue borl state (fst $ head bestRho)
                 --    vVal <- vValue decideVPlusPsi borl state (fst $ head bestV) -- all a have the same V(s,a) value!
                 --    r0Values <- mapM (rValue borl RSmall state . fst) bestV
                 --    let rhoPlusV = rhoVal / (1-gamma0) + vVal
                 --        (posErr,negErr) = (map snd *** map snd) $ partition ((rhoPlusV<) . fst) (zip r0Values (zip eVals bestV))
                 --    return $ map snd $ head $ groupBy (epsCompare (==) `on` fst) $ sortBy (epsCompare compare `on` fst) (if null posErr then negErr else posErr)
                 ----
                 if length bestE == 1 -- Uniform distributed as all actions are considered to have the same value!
                   then return (borl, False, headE bestE)
                   else do
                     r <- liftIO $ randomRIO (0, length bestE - 1)
                     return (borl, False, bestE !! r)
           AlgBORLVOnly {} -> singleValueNextAction EpsilonSensitive (vValue borl state . fst)
           AlgDQN _ cmp -> singleValueNextAction cmp (rValue borl RBig state . fst)
           AlgDQNAvgRewAdjusted {} -> do
             bestV <-
               do vValues <- mapM (vValue borl state . fst) as -- 1. choose highest bias values
                  map snd . maximised <$> liftIO (selFromList $ groupBy (epsCompare (==) `on` fst) $ sortBy (compare `on` fst) (zip vValues as))
             if length bestV == 1
               then return (borl, False, head bestV)
               else do
                 r1Values <- mapM (rValue borl RBig state . fst) bestV -- 2. choose action by epsilon-max R1 (near-Blackwell-optimal algorithm)
                 bestR1ValueActions <- liftIO $ fmap maximised $ selFromList $ groupBy (epsCompare (==) `on` fst) $ sortBy (compare `on` fst) (zip r1Values bestV)
                 let bestR1 = map snd bestR1ValueActions
                 if length bestR1 == 1
                   then return (borl, False, head bestR1)
                   else do
                     r <- liftIO $ randomRIO (0, length bestR1 - 1) --  3. Uniform selection of leftover actions
                     return (borl, False, bestR1 !! r)
  where
    headE []    = error "head: empty input data in nextAction on E Value"
    headE (x:_) = x
    headDqn []    = error "head: empty input data in nextAction on Dqn Value"
    headDqn (x:_) = x
    params' = (borl ^. decayFunction) (borl ^. t) (borl ^. parameters)
    eps = params' ^. epsilon
    explore = params' ^. exploration
    state = borl ^. s
    as = actionsIndexed borl state
    epsCompare = epsCompareWithFactor 1
    epsCompareWithFactor fact = epsCompareWith (fact * eps)
    singleValueNextAction cmp f = do
      rValues <- mapM f as
      let groupValues =
            case cmp of
              EpsilonSensitive -> groupBy (epsCompare (==) `on` fst) . sortBy (compare `on` fst)
              Exact -> groupBy ((==) `on` fst) . sortBy (compare `on` fst)
      bestR <- liftIO $ fmap maximised $ selFromList $ groupValues (zip rValues as)
      if length bestR == 1
        then return (borl, False, snd $ headDqn bestR)
        else do
          r <- liftIO $ randomRIO (0, length bestR - 1)
          return (borl, False, snd $ bestR !! r)


-- | Compare values epsilon-sensitive. Must be used on a sorted list using a standard order.
--
-- > groupBy (epsCompareWith 2 (==)) $ sortBy (compare) [3,5,1]
-- > [[1,3],[5]]
--
epsCompareWith :: (Ord t, Num t) => t -> (t -> t -> p) -> t -> t -> p
epsCompareWith eps f x y
  | abs (x - y) <= eps = f 0 0
  | otherwise = y `f` x
