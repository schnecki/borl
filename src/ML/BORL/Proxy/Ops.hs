{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ExplicitForAll            #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE ScopedTypeVariables       #-}
{-# LANGUAGE TemplateHaskell           #-}
{-# LANGUAGE UndecidableInstances      #-}

module ML.BORL.Proxy.Ops
    ( insert
    , lookupProxy
    , lookupNeuralNetwork
    , lookupNeuralNetworkUnscaled
    , lookupActionsNeuralNetwork
    , lookupActionsNeuralNetworkUnscaled
    , getMinMaxVal
    , mkNNList
    ) where


import           ML.BORL.Calculation
import           ML.BORL.Fork
import           ML.BORL.NeuralNetwork
import           ML.BORL.Proxy.Type
import           ML.BORL.Type
import           ML.BORL.Types                as T
import           ML.BORL.Types

import           Control.Arrow
import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Control.Parallel.Strategies
import           Data.List                    (foldl')
import qualified Data.Map.Strict              as M
import           Data.Singletons.Prelude.List
import           GHC.Generics
import           GHC.TypeLits
import           Grenade


import           Control.Lens
import qualified Data.Map.Strict              as M


-- | Insert (or update) a value.
insert :: forall s . (NFData s, Ord s) => Period -> State s -> ActionIndex -> IsRandomAction -> Reward -> StateNext s -> ReplMemFun s -> Proxies s -> T.MonadBorl (Proxies s, Calculation)
insert period s aNr randAct rew s' getCalc (Proxies pRhoMin pRho pPsiV pV pW pR0 pR1 Nothing) = do
  calc <- getCalc s aNr randAct rew s'
  -- forkMv' <- Simple $ doFork $ P.insert period label vValStateNew mv
  -- mv' <- Simple $ collectForkResult forkMv'

  pRhoMin' <- insertProxy period s aNr (getRhoMinimumVal' calc) pRhoMin `using` rpar
  pRho'    <- insertProxy period s aNr (getRhoVal' calc) pRho           `using` rpar
  pV'      <- insertProxy period s aNr (getVValState' calc) pV          `using` rpar
  pW'      <- insertProxy period s aNr (getWValState' calc) pW          `using` rpar
  pPsiV'   <- insertProxy period s aNr (getPsiVVal' calc) pPsiV         `using` rpar
  pR0'     <- insertProxy period s aNr (getR0ValState' calc) pR0        `using` rpar
  pR1'     <- insertProxy period s aNr (getR1ValState' calc) pR1        `using` rpar
  return (Proxies pRhoMin' pRho' pPsiV' pV' pW' pR0' pR1' Nothing, calc)
insert period s aNr randAct rew s' getCalc pxs@(Proxies pRhoMin pRho pPsiV pV pW pR0 pR1 (Just replMem))
  | period <= fromIntegral (replMem ^. replayMemorySize) - 1 = do
    replMem' <- Simple $ addToReplayMemory period (s, aNr, randAct, rew, s') replMem
    (pxs', calc) <- insert period s aNr randAct rew s' getCalc (replayMemory .~ Nothing $ pxs)
    return (replayMemory ?~ replMem' $ pxs', calc)
  | pV ^?! proxyNNConfig . trainBatchSize == 1 = do
    replMem' <- Simple $ addToReplayMemory period (s, aNr, randAct, rew, s') replMem
    calc <- getCalc s aNr randAct rew s'
    pRho'    <- insertProxy period s aNr (getRhoVal' calc) pRho >>= if isNeuralNetwork pRho then updateNNTargetNet False period else return              `using` rpar
    pRhoMin' <- insertProxy period s aNr (getRhoMinimumVal' calc) pRhoMin >>= if isNeuralNetwork pRhoMin then updateNNTargetNet False period else return `using` rpar
    pV'      <- insertProxy period s aNr (getVValState' calc) pV >>= updateNNTargetNet False period                                                      `using` rpar
    pW'      <- insertProxy period s aNr (getWValState' calc) pW >>= updateNNTargetNet False period                                                      `using` rpar
    pPsiV'   <- insertProxy period s aNr (getPsiVVal' calc) pPsiV >>= updateNNTargetNet False period                                                     `using` rpar
    pR0'     <- insertProxy period s aNr (getR0ValState' calc) pR0 >>= updateNNTargetNet False period                                                    `using` rpar
    pR1'     <- insertProxy period s aNr (getR1ValState' calc) pR1 >>= updateNNTargetNet False period                                                    `using` rpar
    return (Proxies pRhoMin' pRho' pPsiV' pV' pW' pR0' pR1' (Just replMem'), calc) -- avgCalculation (map snd calcs))
  | otherwise = do
    replMem' <- Simple $ addToReplayMemory period (s, aNr, randAct, rew, s') replMem
    calc <- getCalc s aNr randAct rew s'
    let config = pV ^?! proxyNNConfig
    mems <- Simple $ getRandomReplayMemoryElements period (config ^. trainBatchSize) replMem'
    let mkCalc (s, idx, rand, rew, s') = getCalc s idx rand rew s'
    calcs <- parMap rdeepseq force <$> mapM (\m@(s, idx, _, _, _) -> mkCalc m >>= \v -> return ((config ^. toNetInp $ s, idx), v)) mems

    -- let avgCalc = avgCalculation (map snd calcs)
    pRhoMin' <- if isNeuralNetwork pRhoMin then trainBatch (map (second getRhoMinimumVal') calcs) pRhoMin >>= updateNNTargetNet False period `using` rpar
      else insertProxy period s aNr (getRhoMinimumVal' calc) -- avgCalc)
      pRhoMin `using` rpar
    pRho' <- if isNeuralNetwork pRho then trainBatch (map (second getRhoVal') calcs) pRho >>= updateNNTargetNet False period  `using` rpar
      else insertProxy period s aNr (getRhoVal' calc) -- avgCalc)
      pRho `using` rpar
    pV'    <- trainBatch (map (second getVValState') calcs) pV >>= updateNNTargetNet False period   `using` rpar
    pW'    <- trainBatch (map (second getWValState') calcs) pW >>= updateNNTargetNet False period   `using` rpar
    pPsiV' <- trainBatch (map (second getPsiVVal') calcs) pPsiV >>= updateNNTargetNet False period  `using` rpar
    pR0'   <- trainBatch (map (second getR0ValState') calcs) pR0 >>= updateNNTargetNet False period `using` rpar
    pR1'   <- trainBatch (map (second getR1ValState') calcs) pR1 >>= updateNNTargetNet False period `using` rpar
    return (Proxies pRhoMin' pRho' pPsiV' pV' pW' pR0' pR1' (Just replMem'), calc)
            -- avgCalculation (map snd calcs))


-- | Insert a new (single) value to the proxy. For neural networks this will add the value to the startup table. See
-- `trainBatch` to train the neural networks.
insertProxy :: forall s . (NFData s, Ord s) => Period -> State s -> ActionIndex -> Double -> Proxy s -> T.MonadBorl (Proxy s)
insertProxy _ _ _ v (Scalar _) = return $ Scalar v
insertProxy _ s aNr v (Table m) = return $ Table (M.insert (s,aNr) v m)
insertProxy period st idx v px
  | period < fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize) - 1 = return $ proxyNNStartup .~ M.insert (st, idx) v tab $ px
  | period == fromIntegral (px ^?! proxyNNConfig . replayMemoryMaxSize) - 1 = Simple (putStrLn $ "Initializing artificial neural networks: " ++ show (px ^? proxyType)) >> netInit px >>= updateNNTargetNet True period
  | otherwise = trainBatch [((config ^.  toNetInp $ st, idx), v)] px

  where
    netInit = trainMSE (Just 0) (M.toList tab) (config ^. learningParams)
    config = px ^?! proxyNNConfig
    tab = px ^?! proxyNNStartup

-- | Copy the worker net to the target.
updateNNTargetNet :: Bool -> Period -> Proxy s -> T.MonadBorl (Proxy s)
updateNNTargetNet forceReset period px@(Grenade _ netW' tab' tp' config' nrActs)
  | not forceReset && period <= fromIntegral memSize = return px
  | forceReset || (period - fromIntegral memSize - 1) `mod` config' ^. updateTargetInterval == 0 = return $ Grenade netW' netW' tab' tp' config' nrActs
  | otherwise = return px
  where
    memSize = px ^?! proxyNNConfig . replayMemoryMaxSize
updateNNTargetNet forceReset period px@(TensorflowProxy netT' netW' tab' tp' config' nrActs)
  | not forceReset && period <= fromIntegral memSize = return px
  | forceReset || (period - fromIntegral memSize - 1) `mod` config' ^. updateTargetInterval == 0 = do
    copyValuesFromTo netW' netT'
    return $ TensorflowProxy netT' netW' tab' tp' config' nrActs
  | otherwise = return px
  where
    memSize = px ^?! proxyNNConfig . replayMemoryMaxSize
updateNNTargetNet _ _ _ = error "updateNNTargetNet called on non-neural network proxy"


-- | Train the neural network from a given batch. The training instances are Unscaled.
trainBatch :: forall s . [(([Double], ActionIndex), Double)] -> Proxy s -> T.MonadBorl (Proxy s)
trainBatch trainingInstances px@(Grenade netT netW tab tp config nrActs) = do
  let netW' = foldl' (trainGrenade (config ^. learningParams)) netW (map return trainingInstances')
  return $ Grenade netT netW' tab tp config nrActs
  where trainingInstances' = map (second $ scaleValue (getMinMaxVal px)) trainingInstances
trainBatch trainingInstances px@(TensorflowProxy netT netW tab tp config nrActs) = do
  backwardRunRepMemData netW trainingInstances'
  return $ TensorflowProxy netT netW tab tp config nrActs
  where trainingInstances' = map (second $ scaleValue (getMinMaxVal px)) trainingInstances

trainBatch _ _ = error "called trainBatch on non-neural network proxy (programming error)"


-- | Train until MSE hits the value given in NNConfig.
trainMSE :: (NFData k) => Maybe Int -> [((k, ActionIndex), Double)] -> LearningParameters -> Proxy k -> T.MonadBorl (Proxy k)
trainMSE _ _ _ px@Table{} = return px
trainMSE mPeriod dataset lp px@(Grenade _ netW tab tp config nrActs)
  | mse < mseMax = do
      Simple $ putStrLn $ "Final MSE for " ++ show tp ++ ": " ++ show mse
      return px
  | otherwise = do
      when (maybe False ((==0) . (`mod` 100)) mPeriod) $
        Simple $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
      fmap force <$> trainMSE ((+ 1) <$> mPeriod) dataset lp $ Grenade net' net' tab tp config nrActs
  where
    mseMax = config ^. trainMSEMax
    net' = foldl' (trainGrenade lp) netW (zipWith (curry return) kScaled vScaled)
    -- net' = trainGrenade lp netW (zip kScaled vScaled)
    vScaled = map (scaleValue (getMinMaxVal px) . snd) dataset
    vUnscaled = map snd dataset
    kScaled = map (first (config ^. toNetInp) . fst) dataset
    getValue k =
      -- unscaleValue (getMinMaxVal px) $                                                                                 -- scaled or unscaled ones?
      (!!snd k) $ snd $ fromLastShapes netW $ runNetwork netW ((toHeadShapes netW . (config ^. toNetInp) . fst) k)
    mse = 1 / fromIntegral (length dataset) * sum (zipWith (\k v -> (v - getValue k)**2) (map fst dataset) (map (min 1 . max (-1)) vScaled)) -- scaled or unscaled ones?
trainMSE mPeriod dataset lp px@(TensorflowProxy netT netW tab tp config nrActs) =
  let mseMax = config ^. trainMSEMax
      kFullScaled = map (first (map realToFrac . (config ^. toNetInp)) . fst) dataset :: [([Float], ActionIndex)]
      kScaled = map fst kFullScaled
      actIdxs = map snd kFullScaled
      vScaled = map (realToFrac . scaleValue (getMinMaxVal px) . snd) dataset
      vUnscaled = map (realToFrac . snd) dataset
      datasetRepMem = map (first (first (map realToFrac . (config^.toNetInp)))) dataset
  in do current <- forwardRun netW kScaled
        zipWithM_ (backwardRun netW) (map return kScaled) (map return $ zipWith3 replace actIdxs vScaled current)
        -- backwardRunRepMemData netW datasetRepMem
        let forward k = realToFrac <$> lookupNeuralNetworkUnscaled Worker k px -- lookupNeuralNetwork Worker k px -- scaled or unscaled ones?
        mse <- (1 / fromIntegral (length dataset) *) . sum <$> zipWithM (\k vS -> (**2) . (vS -) <$> forward k) (map fst dataset) (map (min 1 . max (-1)) vScaled) -- vUnscaled -- scaled or unscaled ones?
        if realToFrac mse < mseMax
          then Simple $ putStrLn ("Final MSE for " ++ show tp ++ ": " ++ show mse) >> return px
          else do
            when (maybe False ((== 0) . (`mod` 100)) mPeriod) $ do
              void $ saveModelWithLastIO netW -- Save model to ensure correct values when reading from another session
              Simple $ putStrLn $ "Current MSE for " ++ show tp ++ ": " ++ show mse
            trainMSE ((+ 1) <$> mPeriod) dataset lp (TensorflowProxy netT netW tab tp config nrActs)
trainMSE _ _ _ _ = error "trainMSE should not have been callable with this type of proxy. programming error!"


-- | Retrieve a value.
lookupProxy :: (Ord k) => Period -> LookupType -> (k, ActionIndex) -> Proxy k -> T.MonadBorl Double
lookupProxy _ _ _ (Scalar x) = return x
lookupProxy _ _ k (Table m) = return $ M.findWithDefault 0 k m
lookupProxy period lkType k px
  | period <= fromIntegral (config ^. replayMemoryMaxSize) = return $ M.findWithDefault 0 k tab
  | otherwise = lookupNeuralNetwork lkType k px
  where config = px ^?! proxyNNConfig
        tab = px ^?! proxyNNStartup


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupNeuralNetwork :: LookupType -> (k, ActionIndex) -> Proxy k -> T.MonadBorl Double
lookupNeuralNetwork tp k px@Grenade {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork tp k px@TensorflowProxy {} = unscaleValue (getMinMaxVal px) <$> lookupNeuralNetworkUnscaled tp k px
lookupNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"

-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown. The returned value is up-scaled
-- to the original interval before returned.
lookupActionsNeuralNetwork :: LookupType -> k -> Proxy k -> T.MonadBorl [Double]
lookupActionsNeuralNetwork tp k px@Grenade {} = map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px
lookupActionsNeuralNetwork tp k px@TensorflowProxy {} = map (unscaleValue (getMinMaxVal px)) <$> lookupActionsNeuralNetworkUnscaled tp k px
lookupActionsNeuralNetwork _ _ _ = error "lookupNeuralNetwork called on non-neural network proxy"


-- | Retrieve a value from a neural network proxy. For other proxies an error is thrown.
lookupNeuralNetworkUnscaled :: LookupType -> (k, ActionIndex) -> Proxy k -> T.MonadBorl Double
lookupNeuralNetworkUnscaled Worker (st, actIdx) (Grenade _ netW _ _ conf _) = return $ (!!actIdx) $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (conf ^. toNetInp) st)
lookupNeuralNetworkUnscaled Target (st, actIdx) (Grenade netT _ _ _ conf _) = return $ (!!actIdx) $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT $ (conf ^. toNetInp) st)
lookupNeuralNetworkUnscaled Worker (st, actIdx) (TensorflowProxy _ netW _ _ conf _) = realToFrac . (!!actIdx) . head <$> forwardRun netW [map realToFrac $ (conf^. toNetInp) st]
lookupNeuralNetworkUnscaled Target (st, actIdx) (TensorflowProxy netT _ _ _ conf _) = realToFrac . (!!actIdx) . head <$> forwardRun netT [map realToFrac $ (conf^. toNetInp) st]
lookupNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"

-- | Retrieve all action values of a state from a neural network proxy. For other proxies an error is thrown.
lookupActionsNeuralNetworkUnscaled :: forall k . LookupType -> k -> Proxy k -> T.MonadBorl [Double]
lookupActionsNeuralNetworkUnscaled Worker st (Grenade _ netW _ _ conf _) = return $ snd $ fromLastShapes netW $ runNetwork netW (toHeadShapes netW $ (conf ^. toNetInp) st)
lookupActionsNeuralNetworkUnscaled Target st (Grenade netT _ _ _ conf _) = return $ snd $ fromLastShapes netT $ runNetwork netT (toHeadShapes netT $ (conf ^. toNetInp) st)
lookupActionsNeuralNetworkUnscaled Worker st (TensorflowProxy _ netW _ _ conf _) = map realToFrac . head <$> forwardRun netW [map realToFrac $ (conf^. toNetInp) st]
lookupActionsNeuralNetworkUnscaled Target st (TensorflowProxy netT _ _ _ conf _) = map realToFrac . head <$> forwardRun netT [map realToFrac $ (conf^. toNetInp) st]
lookupActionsNeuralNetworkUnscaled _ _ _ = error "lookupNeuralNetworkUnscaled called on non-neural network proxy"


-- | Finds the correct value for scaling.
getMinMaxVal :: Proxy k -> (MinValue,MaxValue)
getMinMaxVal Table{} = error "getMinMaxVal called for Table"
getMinMaxVal p  = case p ^?! proxyType of
  VTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)
  WTable  -> (p ^?! proxyNNConfig.scaleParameters.scaleMinWValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxWValue)
  R0Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR0Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR0Value)
  R1Table -> (p ^?! proxyNNConfig.scaleParameters.scaleMinR1Value, p ^?! proxyNNConfig.scaleParameters.scaleMaxR1Value)
  PsiVTable -> (p ^?! proxyNNConfig.scaleParameters.scaleMinVValue, p ^?! proxyNNConfig.scaleParameters.scaleMaxVValue)


-- | This function loads the model from the checkpoint file and finds then retrieves the data.
mkNNList :: (Ord k, Eq k) => BORL k -> Bool -> Proxy k -> T.MonadBorl [(k, ([(ActionIndex, Double)], [(ActionIndex, Double)]))]
mkNNList borl scaled pr =
  mapM
    (\st -> do
       let fil = actFilt st
           filterActions xs = map (\(_, a, b) -> (a, b)) $ filter (\(f, _, _) -> f) $ zip3 fil [(0 :: Int) ..] xs
       t <-
         if useTable
           then return $ lookupTable scaled st
           else if scaled
                  then lookupActionsNeuralNetwork Target st pr
                  else lookupActionsNeuralNetworkUnscaled Target st pr
       w <-
         if scaled
           then lookupActionsNeuralNetwork Worker st pr
           else lookupActionsNeuralNetworkUnscaled Worker st pr
       return (st, (filterActions t, filterActions w)))
    (conf ^. prettyPrintElems)
  where
    conf =
      case pr of
        Grenade _ _ _ _ conf _         -> conf
        TensorflowProxy _ _ _ _ conf _ -> conf
        _                              -> error "mkNNList called on non-neural network"
    actIdxs = [0 .. _proxyNrActions pr]
    actFilt = borl ^. actionFilter
    useTable = borl ^. t == fromIntegral (_proxyNNConfig pr ^?! replayMemoryMaxSize)
    lookupTable scale st
      | scale = val -- values are being unscaled, thus let table value be unscaled
      | otherwise = map (scaleValue (getMinMaxVal pr)) val
      where
        val = map (\actNr -> M.findWithDefault 0 (st, actNr) (_proxyNNStartup pr)) [0 .. _proxyNrActions pr]
          -- map snd $ M.toList $ M.filterWithKey (\(x, _) _ -> x == st) (_proxyNNStartup pr)
