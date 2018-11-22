{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ExplicitForAll            #-}
{-# LANGUAGE GADTs                     #-}

module ML.BORL.Proxy
    ( Proxy (..)
    , NNConfig (..)
    , insert
    , findWithDefault
    ) where


import           ML.BORL.NeuralNetwork

import           Control.Arrow
import qualified Data.Map.Strict              as M
import           Data.Singletons.Prelude.List
import           GHC.TypeLits
import           Grenade

type Cache k = [(k,Double)]

-- Todo: 2 Networks (target, worker)
data Proxy k = Table !(M.Map k Double)
             | forall layers shapes nrH nrL . (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL) => NN (Network layers shapes) (NNConfig k)

data NNConfig k = NNConfig
  { toNetInp       :: k -> [Double]
  , cache          :: Cache k
  , trainBatchSize :: Int
  , learninParams  :: LearningParameters
  }

-- | Insert (or update) a value.
insert :: (Ord k) => k -> Double -> Proxy k -> Proxy k
insert k v (Table m)                        = Table (M.insert k v m)
insert k v (NN net (NNConfig toInp chs bs lp)) = checkTrainBatchsize ((k, v) : chs)
  where
    checkTrainBatchsize cache'
      | length cache' >= bs = NN (trainNetwork lp net (map (first toInp) cache')) (NNConfig toInp [] bs lp)
      | otherwise = NN net (NNConfig toInp cache' bs lp)


-- | Retrieve a value.
findWithDefault :: (Ord k) => Double -> k -> Proxy k -> Double
findWithDefault def k (Table m) = M.findWithDefault def k m
findWithDefault _ k (NN net conf) = head $ snd $ fromLastShapes net $ runNetwork net (toHeadShapes net $ toNetInp conf k)
