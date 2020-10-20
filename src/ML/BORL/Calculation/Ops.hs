{-# LANGUAGE CPP                 #-}
{-# LANGUAGE ScopedTypeVariables #-}
module ML.BORL.Calculation.Ops
    ( mkCalculation
    , rhoValueWith
    , rValue
    , rValueAgentWith
    , rValueWith
    , rValueNoUnscaleWith
    , eValue
    , eValueFeat
    , eValueAvgCleaned
    , eValueAvgCleanedAgent
    , eValueAvgCleanedFeat
    , vValue
    , vValueAgentWith
    , vValueWith
    , vValueNoUnscaleWith
    , wValueFeat
    , rhoValue
    , rhoValueAgentWith
    , overEstimateRho
    , RSize (..)
    , expSmthPsi
    ) where

import           Control.Lens
import           Control.Monad.IO.Class
import qualified Data.Vector.Storable as V
import           Control.Parallel.Strategies    hiding (r0)
import           Data.Maybe                     (fromMaybe)
import           Data.List (zipWith6,zipWith5,zipWith4)
import           Control.DeepSeq
import Control.Applicative ((<|>))

import           ML.BORL.Algorithm
import           ML.BORL.Calculation.Type
import           ML.BORL.Decay                  (decaySetup)
import           ML.BORL.NeuralNetwork.NNConfig
import           ML.BORL.Parameters
import           ML.BORL.Properties
import           ML.BORL.Settings
import           ML.BORL.Proxy                  as P
import           ML.BORL.Type
import           ML.BORL.Types

import Debug.Trace

#ifdef DEBUG
import Prelude hiding (maximum, minimum)
import qualified Prelude (maximum, minimum)
import qualified Data.List
maximumBy, minimumBy :: (a -> a -> Ordering) -> [a] -> a
maximumBy _ [] = error "empty input to maximumBy in ML.BORL.Calculation.Ops"
maximumBy f xs = Data.List.maximumBy f xs
minimumBy _ [] = error "empty input to minimumBy in ML.BORL.Calculation.Ops"
minimumBy f xs = Data.List.minimumBy f xs
maximum, minimum :: (Ord a) => [a] -> a
maximum [] = error "empty input to maximum in ML.BORL.Calculation.Ops"
maximum xs = Data.List.maximum xs
minimum [] = error "empty input to minimum in ML.BORL.Calculation.Ops"
minimum xs = Data.List.minimum xs
#else
import           Data.List                      (maximumBy, minimumBy)
#endif


-- | Used to select a discount factor.
data RSize
  = RSmall
  | RBig


expSmthPsi :: Float
expSmthPsi = 0.001

-- expSmthReward :: Float
-- expSmthReward = 0.001


keepXLastValues :: Int
keepXLastValues = 100

mkCalculation ::
     (MonadIO m)
  => BORL s as
  -> (StateFeatures, FilteredActionIndices) -- ^ State features and filtered actions for each agent
  -> ActionChoice -- ^ ActionIndex for each agent
  -> RewardValue
  -> (StateNextFeatures, FilteredActionIndices) -- ^ State features and filtered actions for each agent
  -> EpisodeEnd
  -> ExpectedValuationNext
  -> m (Calculation, ExpectedValuationNext)
mkCalculation borl state as reward stateNext episodeEnd =
  mkCalculation' borl state as reward stateNext episodeEnd (borl ^. algorithm)

ite :: Bool -> p -> p -> p
ite True thenPart _  = thenPart
ite False _ elsePart = elsePart
{-# INLINE ite #-}

rhoMinimumState' :: BORL s as -> Value -> Value
rhoMinimumState' borl rhoVal' = mapValue go rhoVal'
  where
    go v =
      case borl ^. objective of
        Maximise
          | v >= 0 -> max (v - 2) (0.975 * v)
          | otherwise -> max (v - 2) (1.025 * v)
        Minimise
          | v >= 0 -> min (v + 2) (1.025 * v)
          | otherwise -> min (v + 2) (0.975 * v)

-- | Get an exponentially smoothed parameter. Due to lazy evaluation the calculation for the other parameters are
-- ignored!
getExpSmthParam :: BORL s as -> ((Proxy -> Const Proxy Proxy) -> Proxies -> Const Proxy Proxies) -> Getting Float (Parameters Float) Float -> Float
getExpSmthParam borl p param
  | isANN = 1
  | otherwise = params' ^. param
  where
    isANN = P.isNeuralNetwork px && borl ^. t >= px ^?! proxyNNConfig . replayMemoryMaxSize
    px = borl ^. proxies . p
    params' = decayedParameters borl

-- | Overestimates the average reward. This ensures that we constantly aim for better policies.
overEstimateRho :: BORL s as -> Float -> Float
overEstimateRho borl rhoVal = max' (max' expSmthRho rhoVal) (rhoVal + 0.1 * diff)
  where
    expSmthRho = borl ^. expSmoothedReward
    diff = rhoVal - expSmthRho
    max' :: (Ord x) => x -> x -> x
    max' =
      case borl ^. objective of
        Maximise -> max
        Minimise -> min

mkCalculation' ::
     (MonadIO m)
  => BORL s as
  -> (StateFeatures, FilteredActionIndices)
  -> ActionChoice
  -> RewardValue
  -> (StateNextFeatures, FilteredActionIndices)
  -> EpisodeEnd
  -> Algorithm NetInputWoAction
  -> ExpectedValuationNext
  -> m (Calculation, ExpectedValuationNext)
mkCalculation' borl (state, stateActIdxes) as reward (stateNext, stateNextActIdxes) episodeEnd (AlgBORL ga0 ga1 avgRewardType mRefState) expValStateNext = do
  let params' = decayedParameters borl
  let aNr = map snd as
      randomAction = any fst as
  let isRefState = mRefState == Just (state, aNr)
  let alp = getExpSmthParam borl rho alpha
      bta = getExpSmthParam borl v beta
      dltW = getExpSmthParam borl w delta
      gamR0 = getExpSmthParam borl r0 gamma
      gamR1 = getExpSmthParam borl r1 gamma
      alpRhoMin = getExpSmthParam borl rhoMinimum alphaRhoMin
      xiVal = params' ^. xi
      zetaVal = params' ^. zeta
      period = borl ^. t
      (psiValRho, psiValV, psiValW) = borl ^. psis -- exponentially smoothed Psis
  let agents = borl ^. settings . independentAgents
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
  let label = (state, aNr)
      epsEnd
        | episodeEnd = 0
        | otherwise = 1
      randAct
        | randomAction = 0
        | otherwise = 1
      nonRandAct
        | learnFromRandom = 1
        | otherwise = 1 - randAct
  let expSmth
        | learnFromRandom = expSmthPsi
        | otherwise = randAct * expSmthPsi
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _ -> take keepXLastValues $ reward : borl ^. lastRewards
  vValState <- vValueWith Worker borl state aNr `using` rpar
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  vValStateNext <- vStateValueWith Target borl (stateNext, stateNextActIdxes) `using` rpar
  rhoVal <- rhoValueWith Worker borl state aNr `using` rpar
  wValState <- wValueFeat borl state aNr `using` rpar
  wValStateNext <- wStateValue borl (stateNext, stateNextActIdxes) `using` rpar
  psiVState <- P.lookupProxy period Worker label (borl ^. proxies . psiV) `using` rpar
  psiWState <- P.lookupProxy period Worker label (borl ^. proxies . psiW) `using` rpar
  r0ValState <- rValueWith Worker borl RSmall state aNr `using` rpar
  r0ValStateNext <- rStateValueWith Target borl RSmall (stateNext, stateNextActIdxes) `using` rpar
  r1ValState <- rValueWith Worker borl RBig state aNr `using` rpar
  r1ValStateNext <- rStateValueWith Target borl RBig (stateNext, stateNextActIdxes) `using` rpar
  -- Rho
  rhoState <-
    case avgRewardType of
      Fixed x -> return $ toValue agents x
      ByMovAvg l
        | isUnichain borl -> return $ toValue agents $ sum lastRews' / fromIntegral l
      ByMovAvg _ -> error "ByMovAvg is not allowed in multichain setups"
      ByReward -> return $ toValue agents reward
      ByStateValues -> return $ reward .+ vValStateNext - vValState
      ByStateValuesAndReward ratio decay -> return $ ratio' .* (reward .+ vValStateNext - vValState) +. (1 - ratio') * reward
        where ratio' = decaySetup decay period ratio
  let maxOrMin =
        case borl ^. objective of
          Maximise -> max
          Minimise -> min
  let rhoVal'
        | randomAction && not learnFromRandom = rhoVal
        | otherwise =
          zipWithValue maxOrMin rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x -> toValue agents x
            _ -> (1 - alp) .* rhoVal + alp .* rhoState
  -- RhoMin
  let rhoMinimumVal'
        | randomAction = rhoMinimumState
        | otherwise = zipWithValue maxOrMin rhoMinimumState $ (1 - alpRhoMin) .* rhoMinimumState + alpRhoMin .* rhoMinimumState' borl rhoVal'
  -- PsiRho (should converge to 0)
  psiRho <- ite (isUnichain borl) (return $ rhoVal' - rhoVal) (subtract rhoVal' <$> rhoStateValue borl (stateNext, stateNextActIdxes))
  -- V
  let rhoValOverEstimated = mapValue (overEstimateRho borl) rhoVal'
  let vValState' = (1 - bta) .* vValState + bta .* (reward .- rhoValOverEstimated + epsEnd .* vValStateNext + nonRandAct .* (psiVState + zetaVal .* psiWState))
      psiV = reward .+ vValStateNext - rhoValOverEstimated - vValState' -- should converge to 0
      psiVState' = (1 - xiVal * bta) .* psiVState + bta * xiVal .* psiV
  -- LastVs
  let lastVs' = take keepXLastValues $ vValState' : borl ^. lastVValues
  -- W
  let wValState'
        | isRefState = 0
        | otherwise = (1 - dltW) .* wValState + dltW .* (-vValState' + epsEnd .* wValStateNext + nonRandAct .* psiWState)
      psiW = wValStateNext - vValState' - wValState'
      psiWState'
        | isRefState = 0
        | otherwise = (1 - xiVal * dltW) .* psiWState + dltW * xiVal .* psiW
  -- R0/R1
  let r0ValState' = (1 - gamR0) .* r0ValState + gamR0 .* (reward .+ epsEnd * ga0 .* r0ValStateNext)
  let r1ValState' = (1 - gamR1) .* r1ValState + gamR1 .* (reward .+ epsEnd * ga1 .* r1ValStateNext)
  -- Psis Scalar calues for output only
  let psiValRho' = (1 - expSmth) .* psiValRho + expSmth .* abs psiRho
  let psiValV' = (1 - expSmth) .* psiValV + expSmth .* abs psiVState'
  let psiValW' = (1 - expSmth) .* psiValW + expSmth .* abs psiWState'
  return $
    ( Calculation
        { getRhoMinimumVal' = Just rhoMinimumVal'
        , getRhoVal' = Just rhoVal'
        , getPsiVValState' = Just psiVState'
        , getVValState' = Just vValState'
        , getPsiWValState' = Just psiWState' -- $ ite isRefState 0 psiWState'
        , getWValState' = Just $ ite isRefState 0 wValState'
        , getR0ValState' = Just r0ValState'
        , getR1ValState' = Just r1ValState'
        , getPsiValRho' = Just psiValRho'
        , getPsiValV' = Just psiValV'
        , getPsiValW' = Just psiValW'
        , getLastVs' = Just (force lastVs')
        , getLastRews' = force lastRews'
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = ite (randomAction && not learnFromRandom) (borl ^. expSmoothedReward) ((1 - alp) * borl ^. expSmoothedReward + alp * reward)
        }
    , ExpectedValuationNext
        { getExpectedValStateNextRho = Nothing
        , getExpectedValStateNextV = error "N-Step not implemented in AlgBORL"
        , getExpectedValStateNextW = Nothing
        , getExpectedValStateNextR0 = Nothing
        , getExpectedValStateNextR1 = Nothing
        })

mkCalculation' borl sa@(state, _) as reward (stateNext, stateNextActIdxes) episodeEnd (AlgDQNAvgRewAdjusted ga0 ga1 avgRewardType) expValStateNext = do
  let aNr = map snd as
      randomAction = any fst as
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr
  rhoVal <- rhoValueWith Worker borl state aNr
  r0ValState <- rValueWith Worker borl RSmall state aNr `using` rpar
  r0StateNext <- rStateValueWith Target borl RSmall (stateNext, stateNextActIdxes) `using` rpar
  r1ValState <- rValueWith Worker borl RBig state aNr `using` rpar
  r1StateNext <- rStateValueWith Target borl RBig (stateNext, stateNextActIdxes) `using` rpar
  r1StateNextWorker <- rStateValueWith Worker borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let alp = getExpSmthParam borl rho alpha
      alpRhoMin = getExpSmthParam borl rhoMinimum alphaRhoMin
      gam = getExpSmthParam borl r1 gamma
  let agents = borl ^. settings . independentAgents
  let params' = decayedParameters borl
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _ -> take keepXLastValues $ reward : borl ^. lastRewards
  -- Rho
  rhoState <-
    case avgRewardType of
      Fixed x -> return $ toValue agents x
      ByMovAvg l -> return $ toValue agents $ sum lastRews' / fromIntegral l
      ByReward -> return $ toValue agents reward
      ByStateValues -> return $ reward .+ r1StateNextWorker - r1ValState
      ByStateValuesAndReward ratio decay -> return $ ratio' .* (reward .+ r1StateNextWorker - r1ValState) +. (1 - ratio') * reward
        where ratio' = decaySetup decay (borl ^. t) ratio
  let maxOrMin =
        case borl ^. objective of
          Maximise -> max
          Minimise -> min
  let rhoVal'
        | randomAction && not learnFromRandom = zipWithValue maxOrMin rhoMinimumState rhoVal
        | otherwise =
          zipWithValue maxOrMin rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x -> toValue agents x
            _ -> (1 - alp) .* rhoVal + alp .* rhoState
      rhoValOverEstimated = mapValue (overEstimateRho borl) rhoVal'
  -- RhoMin
  let rhoMinimumVal'
        | randomAction = rhoMinimumState
        | otherwise = zipWithValue maxOrMin rhoMinimumState $ (1 - alpRhoMin) .* rhoMinimumState + alpRhoMin .* rhoMinimumState' borl rhoVal'
  let expStateNextValR0
        | randomAction = r0StateNext
        | otherwise = fromMaybe r0StateNext (getExpectedValStateNextR0 expValStateNext)
      expStateNextValR1
        | randomAction = r1StateNext
        | otherwise = fromMaybe r1StateNext (getExpectedValStateNextR1 expValStateNext)
      expStateValR0 = reward .- rhoValOverEstimated + ga0 * epsEnd .* expStateNextValR0
      expStateValR1 = reward .- rhoValOverEstimated + ga1 * epsEnd .* expStateNextValR1
  let r0ValState' = (1 - gam) .* r0ValState + gam .* expStateValR0
  let r1ValState' = (1 - gam) .* r1ValState + gam .* expStateValR1
  let expSmthRewRate = min alp 0.001
  return
    ( Calculation
        { getRhoMinimumVal' = Just rhoMinimumVal'
        , getRhoVal' = Just rhoVal'
        , getPsiVValState' = Nothing
        , getVValState' = Nothing
        , getPsiWValState' = Nothing
        , getWValState' = Nothing
        , getR0ValState' = Just r0ValState' -- gamma middle/low
        , getR1ValState' = Just r1ValState' -- gamma High
        , getPsiValRho' = Nothing
        , getPsiValV' = Nothing
        , getPsiValW' = Nothing
        , getLastVs' = Nothing
        , getLastRews' = force lastRews'
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = ite (randomAction && not learnFromRandom) (borl ^. expSmoothedReward) ((1 - expSmthRewRate) * borl ^. expSmoothedReward + expSmthRewRate * reward)
        }
    , ExpectedValuationNext
        { getExpectedValStateNextRho = Nothing
        , getExpectedValStateNextV = Nothing
        , getExpectedValStateNextW = Nothing
        , getExpectedValStateNextR0 = Just expStateValR0
        , getExpectedValStateNextR1 = Just expStateValR1
        })
mkCalculation' borl (state, _) as reward (stateNext, stateNextActIdxes) episodeEnd (AlgBORLVOnly avgRewardType mRefState) expValStateNext = do
  let aNr = map snd as
      randomAction = any fst as
  let alp = getExpSmthParam borl rho alpha
      alpRhoMin = getExpSmthParam borl rhoMinimum alphaRhoMin
      bta = getExpSmthParam borl v beta
      agents = borl ^. settings . independentAgents
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
      params' = decayedParameters borl
  rhoVal <- rhoValueWith Worker borl state aNr `using` rpar
  vValState <- vValueWith Worker borl state aNr `using` rpar
  vValStateNext <- vStateValueWith Target borl (stateNext, stateNextActIdxes) `using` rpar
  let lastRews' =
        case avgRewardType of
          ByMovAvg movAvgLen -> take movAvgLen $ reward : borl ^. lastRewards
          _ -> take keepXLastValues $ reward : borl ^. lastRewards
  rhoMinimumState <- rhoMinimumValueFeat borl state aNr `using` rpar
  rhoState <-
    case avgRewardType of
      Fixed x -> return $ toValue agents x
      ByMovAvg _ -> return $ toValue agents $ sum lastRews' / fromIntegral (length lastRews')
      ByReward -> return $ toValue agents reward
      ByStateValues -> return $ reward .+ vValStateNext - vValState
      ByStateValuesAndReward ratio decay -> return $ ratio' .* (reward .+ vValStateNext - vValState) +. (1 - ratio') * reward
        where ratio' = decaySetup decay (borl ^. t) ratio
  let maxOrMin =
        case borl ^. objective of
          Maximise -> max
          Minimise -> min
  let rhoVal'
        | randomAction = rhoVal
        | otherwise =
          zipWithValue maxOrMin rhoMinimumState $
          case avgRewardType of
            ByMovAvg _ -> rhoState
            Fixed x -> toValue agents x
            _ -> (1 - alp) .* rhoVal + alp .* rhoState
      rhoValOverEstimated = mapValue (overEstimateRho borl) rhoVal'
  let rhoMinimumVal'
        | randomAction = rhoMinimumState
        | otherwise = zipWithValue maxOrMin rhoMinimumState $ (1 - alpRhoMin) .* rhoMinimumState + alpRhoMin .* rhoMinimumState' borl rhoVal'
  let expStateNextValV
        | randomAction = epsEnd * vValStateNext
        | otherwise = fromMaybe (epsEnd * vValStateNext) (getExpectedValStateNextV expValStateNext)
      expStateValV = reward .- rhoValOverEstimated + expStateNextValV
  let vValState' = (1 - bta) .* vValState + bta .* (reward .- rhoValOverEstimated + expStateValV)
  let lastVs' = take keepXLastValues $ vValState' : borl ^. lastVValues
  return $
    ( Calculation
        { getRhoMinimumVal' = Just rhoMinimumVal'
        , getRhoVal' = Just rhoVal'
        , getPsiVValState' = Nothing
        , getVValState' = Just $ ite (mRefState == Just (state, aNr)) 0 vValState'
        , getPsiWValState' = Nothing
        , getWValState' = Nothing
        , getR0ValState' = Nothing
        , getR1ValState' = Nothing
        , getPsiValRho' = Nothing
        , getPsiValV' = Nothing
        , getPsiValW' = Nothing
        , getLastVs' = Just $ force lastVs'
        , getLastRews' = force lastRews'
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = ite (randomAction && not learnFromRandom) (borl ^. expSmoothedReward) ((1 - alp) * borl ^. expSmoothedReward + alp * reward)
        }
    , ExpectedValuationNext
        { getExpectedValStateNextRho = Nothing
        , getExpectedValStateNextV = Just expStateValV
        , getExpectedValStateNextW = Nothing
        , getExpectedValStateNextR0 = Nothing
        , getExpectedValStateNextR1 = Nothing
        })

mkCalculation' borl (state, _) as reward (stateNext, stateNextActIdxes) episodeEnd (AlgDQN ga _) expValStateNext = do
  let aNr = map snd as
      randomAction = any fst as
  let gam = getExpSmthParam borl r1 gamma
  let epsEnd
        | episodeEnd = 0
        | otherwise = 1
  let learnFromRandom = params' ^. exploration > params' ^. learnRandomAbove
      params' = decayedParameters borl
  let lastRews' = take keepXLastValues $ reward : borl ^. lastRewards
  r1ValState <- rValueWith Worker borl RBig state aNr `using` rpar
  r1StateNext <- rStateValueWith Target borl RBig (stateNext, stateNextActIdxes) `using` rpar
  let expStateNextValR1
        | randomAction = epsEnd * r1StateNext
        | otherwise = fromMaybe (epsEnd * r1StateNext) (getExpectedValStateNextR1 expValStateNext)
      expStateValR1 = reward .+ ga .* expStateNextValR1
  let r1ValState' = (1 - gam) .* r1ValState + gam .* (reward .+ epsEnd * expStateValR1)
  return
    ( emptyCalculation
        { getR1ValState' = Just r1ValState'
        , getLastRews' = force lastRews'
        , getEpisodeEnd = episodeEnd
        , getExpSmoothedReward' = ite (randomAction && not learnFromRandom) (borl ^. expSmoothedReward) ((1-gam) * borl ^. expSmoothedReward + gam * reward)
        }
    , emptyExpectedValuationNext {getExpectedValStateNextR1 = Just expStateValR1})

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoMinimumValue :: (MonadIO m) => BORL s as -> State s -> [ActionIndex] -> m Value
rhoMinimumValue borl state = rhoMinimumValueWith Worker borl (ftExt state)
  where
    ftExt = borl ^. featureExtractor

rhoMinimumValueFeat :: (MonadIO m) => BORL s as -> StateFeatures -> [ActionIndex] -> m Value
rhoMinimumValueFeat = rhoMinimumValueWith Worker

rhoMinimumValueWith :: (MonadIO m) => LookupType -> BORL s as -> StateFeatures -> [ActionIndex] -> m Value
rhoMinimumValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state,a) (borl ^. proxies.rhoMinimum)

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoValue :: (MonadIO m) => BORL s as -> State s -> [ActionIndex] -> m Value
rhoValue borl s = rhoValueWith Worker borl (ftExt s)
  where
    ftExt = borl ^. featureExtractor

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
rhoValueAgentWith :: (MonadIO m) => LookupType -> BORL s as -> P.AgentNumber -> State s -> ActionIndex -> m Float
rhoValueAgentWith lkTp borl agent s a = P.lookupProxyAgent (borl ^. t) lkTp agent (ftExt s, a) (borl ^. proxies . rho)
  where
    ftExt = borl ^. featureExtractor


rhoValueWith :: (MonadIO m) => LookupType -> BORL s as -> StateFeatures -> [ActionIndex] -> m Value
rhoValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state,a) (borl ^. proxies.rho)

rhoStateValue :: (MonadIO m) => BORL s as -> (StateFeatures, FilteredActionIndices) -> m Value
rhoStateValue borl (state, actIdxes) =
  case borl ^. proxies . rho of
    Scalar r -> return $ AgentValue $ V.toList r
    _ -> reduceValues maxOrMin <$> lookupState Target (state, actIdxes) (borl ^. proxies . rho)
      -- V.mapM (rhoValueWith Target borl state) actIdxes
  where
    maxOrMin =
      case borl ^. objective of
        Maximise -> V.maximum
        Minimise -> V.minimum

-- | Bias value from Worker net.
vValue :: (MonadIO m) => BORL s as -> State s -> [ActionIndex] -> m Value
vValue borl s = vValueWith Worker borl (ftExt s)
  where
    ftExt = borl ^. featureExtractor

-- | Expected average value of state-action tuple, that is y_{-1}(s,a).
vValueAgentWith :: (MonadIO m) => LookupType -> BORL s as -> P.AgentNumber -> State s -> ActionIndex -> m Float
vValueAgentWith lkTp borl agent s a = P.lookupProxyAgent (borl ^. t) lkTp agent (ftExt s, a) (borl ^. proxies . v)
  where
    ftExt = borl ^. featureExtractor


-- | Get bias value from specified net and with features.
vValueWith :: (MonadIO m) => LookupType -> BORL s as -> StateFeatures -> [ActionIndex] -> m Value
vValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . v)

-- | For DEBUGGING only!
vValueNoUnscaleWith :: (MonadIO m) => LookupType -> BORL s as -> StateFeatures -> [ActionIndex] -> m Value
vValueNoUnscaleWith lkTp borl state a = P.lookupProxyNoUnscale (borl ^. t) lkTp (state, a) (borl ^. proxies . v)


-- | Get maximum bias value of state of specified net.
vStateValueWith :: (MonadIO m) => LookupType -> BORL s as -> (StateFeatures, FilteredActionIndices) -> m Value
vStateValueWith lkTp borl (state, asIdxes) = reduceValues maxOrMin <$> lookupState lkTp (state, asIdxes) (borl ^. proxies . v)
  -- V.mapM (vValueWith lkTp borl state) asIdxes
  where
    maxOrMin =
      case borl ^. objective of
        Maximise -> V.maximum
        Minimise -> V.minimum


-- psiVValueWith :: (MonadIO m) => LookupType -> BORL s as -> StateFeatures -> ActionIndex -> m Float
-- psiVValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . psiV)

wValue :: (MonadIO m) => BORL s as -> State s -> [ActionIndex] -> m Value
wValue borl state a = wValueWith Worker borl (ftExt state) a
  where
    ftExt = borl ^. featureExtractor


wValueFeat :: (MonadIO m) => BORL s as -> StateFeatures -> [ActionIndex] -> m Value
wValueFeat = wValueWith Worker

wValueWith :: (MonadIO m) => LookupType -> BORL s as -> StateFeatures -> [ActionIndex] -> m Value
wValueWith lkTp borl state a = P.lookupProxy (borl ^. t) lkTp (state, a) (borl ^. proxies . w)

wStateValue :: (MonadIO m) => BORL s as -> (StateFeatures, FilteredActionIndices) -> m Value
wStateValue borl (state, asIdxes) = reduceValues maxOrMin <$> lookupState Target (state, asIdxes) (borl ^. proxies . w)
  -- V.mapM (wValueWith Target borl state) asIdxes
  where
    maxOrMin =
      case borl ^. objective of
        Maximise -> V.maximum
        Minimise -> V.minimum


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValue :: (MonadIO m) => BORL s as -> RSize -> State s -> [ActionIndex] -> m Value
rValue borl size s aNr = rValueWith Worker borl size (ftExt s) aNr
  where ftExt = borl ^. featureExtractor

-- | Calculates the expected discounted value with the provided gamma (small/big).
rValueAgentWith :: (MonadIO m) => LookupType -> BORL s as -> RSize -> AgentNumber -> State s -> ActionIndex -> m Float
rValueAgentWith lkTp borl size agent s aNr = P.lookupProxyAgent (borl ^. t) lkTp agent (ftExt s, aNr) mr
  where
    ftExt = borl ^. featureExtractor
    mr =
      case size of
        RSmall -> borl ^. proxies . r0
        RBig -> borl ^. proxies . r1


-- | Calculates the expected discounted value with the provided gamma (small/big).
rValueWith :: (MonadIO m) => LookupType -> BORL s as -> RSize -> StateFeatures -> [ActionIndex] -> m Value
rValueWith lkTp borl size state a = P.lookupProxy (borl ^. t) lkTp (state, a) mr
  where
    mr =
      case size of
        RSmall -> borl ^. proxies.r0
        RBig   -> borl ^. proxies.r1

-- | For DEBUGGING only! Same as rValueWith but without unscaling.
rValueNoUnscaleWith :: (MonadIO m) => LookupType -> BORL s as -> RSize -> StateFeatures -> [ActionIndex] -> m Value
rValueNoUnscaleWith lkTp borl size state a = P.lookupProxyNoUnscale (borl ^. t) lkTp (state, a) mr
  where
    mr =
      case size of
        RSmall -> borl ^. proxies.r0
        RBig   -> borl ^. proxies.r1


rStateValueWith :: (MonadIO m) => LookupType -> BORL s as -> RSize -> (StateFeatures, FilteredActionIndices) -> m Value
rStateValueWith lkTp borl size (state, actIdxes) = reduceValues maxOrMin <$> lookupState lkTp (state, actIdxes) mr
  -- V.mapM (rValueWith lkTp borl size state) actIdxes
  where
    maxOrMin =
      case borl ^. objective of
        Maximise -> V.maximum
        Minimise -> V.minimum
    mr =
      case size of
        RSmall -> borl ^. proxies . r0
        RBig -> borl ^. proxies . r1

-- | Calculates the difference between the expected discounted values: e_gamma0 - e_gamma1 (Small-Big).
eValue :: (MonadIO m) => BORL s as -> s -> [ActionIndex] -> m Value
eValue borl state act = eValueFeat borl (borl ^. featureExtractor $ state, act)

-- | Calculates the difference between the expected discounted values: e_gamma0 - e_gamma1 (Small-Big).
eValueFeat :: (MonadIO m) => BORL s as -> (StateFeatures, [ActionIndex]) -> m Value
eValueFeat borl (stateFeat, act) = do
  big <- rValueWith Target borl RBig stateFeat act
  small <- rValueWith Target borl RSmall stateFeat act
  return $ small - big

-- | Calculates the difference between the expected discounted values: e_gamma1 - e_gamma0 - avgRew * (1/(1-gamma1)+1/(1-gamma0)).
eValueAvgCleanedFeat :: (MonadIO m) => BORL s as -> StateFeatures -> [ActionIndex] -> m Value
eValueAvgCleanedFeat borl state act =
  case borl ^. algorithm of
    AlgBORL gamma0 gamma1 _ _ -> avgRewardClean gamma0 gamma1
    AlgDQNAvgRewAdjusted gamma0 gamma1 _ -> avgRewardClean gamma0 gamma1
    _ -> error "eValueAvgCleaned can only be used with AlgBORL in Calculation.Ops"
  where
    avgRewardClean gamma0 gamma1 = do
      rBig <- rValueWith Target borl RBig state act
      rSmall <- rValueWith Target borl RSmall state act
      rhoVal <- rhoValueWith Worker borl state act
      return $ rBig - rSmall - rhoVal *. (1 / (1 - gamma1) - 1 / (1 - gamma0))
    agents = borl ^. settings . independentAgents


-- | Calculates the difference between the expected discounted values: e_gamma1 - e_gamma0 - avgRew * (1/(1-gamma1)+1/(1-gamma0)).
eValueAvgCleaned :: (MonadIO m) => BORL s as -> s -> [ActionIndex] -> m Value
eValueAvgCleaned borl state = eValueAvgCleanedFeat borl sFeat
  where
    sFeat = (borl ^. featureExtractor) state

-- | Calculates the difference between the expected discounted values: e_gamma1 - e_gamma0 - avgRew * (1/(1-gamma1)+1/(1-gamma0)).
eValueAvgCleanedAgent :: (MonadIO m) => BORL s as -> AgentNumber -> s -> ActionIndex -> m Float
eValueAvgCleanedAgent borl agent state act =
  case borl ^. algorithm of
    AlgBORL gamma0 gamma1 _ _ -> avgRewardClean gamma0 gamma1
    AlgDQNAvgRewAdjusted gamma0 gamma1 _ -> avgRewardClean gamma0 gamma1
    _ -> error "eValueAvgCleaned can only be used with AlgBORL in Calculation.Ops"
  where
    avgRewardClean gamma0 gamma1 = do
      rBig <- rValueAgentWith Target borl RBig agent state act
      rSmall <- rValueAgentWith Target borl RSmall agent state act
      rhoVal <- rhoValueAgentWith Worker borl agent state act
      return $ rBig - rSmall - rhoVal * (1 / (1 - gamma1) - 1 / (1 - gamma0))
    agents = borl ^. settings . independentAgents
