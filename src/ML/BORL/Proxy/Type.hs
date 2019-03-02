{-# LANGUAGE DataKinds                 #-}
{-# LANGUAGE DeriveAnyClass            #-}
{-# LANGUAGE DeriveGeneric             #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE GADTs                     #-}
{-# LANGUAGE TemplateHaskell           #-}
{-# LANGUAGE TypeFamilies              #-}

module ML.BORL.Proxy.Type where

import           ML.BORL.NeuralNetwork
import           ML.BORL.Types                as T

import           Control.DeepSeq
import           Control.Lens
import qualified Data.Map.Strict              as M
import           Data.Singletons.Prelude.List
import           GHC.Generics
import           GHC.TypeLits
import           Grenade


-- | Type of approximation (needed for scaling of values).
data ProxyType
  = VTable
  | WTable
  | R0Table
  | R1Table
  | PsiVTable
  deriving (Show, NFData, Generic)

data LookupType = Target | Worker


data Proxy s = Scalar           -- ^ Combines multiple proxies in one for performance benefits.
               { _proxyScalar :: Double
               }
             | Table            -- ^ Representation using a table.
               { _proxyTable :: !(M.Map (s,ActionIndex) Double)
               }
             | forall nrL nrH shapes layers. (KnownNat nrH, Head shapes ~ 'D1 nrH, KnownNat nrL, Last shapes ~ 'D1 nrL, NFData (Tapes layers shapes), NFData (Network layers shapes)) =>
                Grenade         -- ^ Use Grenade neural networks.
                { _proxyNNTarget  :: !(Network layers shapes)
                , _proxyNNWorker  :: !(Network layers shapes)
                , _proxyNNStartup :: !(M.Map (s,ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig s)
                , _proxyNrActions :: !Int
                }
             | TensorflowProxy  -- ^ Use Tensorflow neural networks.
                { _proxyTFTarget  :: TensorflowModel'
                , _proxyTFWorker  :: TensorflowModel'
                , _proxyNNStartup :: !(M.Map (s,ActionIndex) Double)
                , _proxyType      :: !ProxyType
                , _proxyNNConfig  :: !(NNConfig s)
                , _proxyNrActions :: !Int
                }
makeLenses ''Proxy

instance (NFData s) => NFData (Proxy s) where
  rnf (Table x)           = rnf x
  rnf (Grenade t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (TensorflowProxy t w tab tp cfg nrActs) = rnf t `seq` rnf w `seq` rnf tab `seq` rnf tp `seq` rnf cfg `seq` rnf nrActs
  rnf (Scalar x) = rnf x


isNeuralNetwork :: Proxy s -> Bool
isNeuralNetwork Grenade{}         = True
isNeuralNetwork TensorflowProxy{} = True
isNeuralNetwork _                 = False


data Proxies s = Proxies        -- ^ This data type holds all data for BORL.
  { _rhoMinimum   :: !(Proxy s)
  , _rho          :: !(Proxy s)
  , _psiV         :: !(Proxy s)
  , _v            :: !(Proxy s)
  , _w            :: !(Proxy s)
  , _r0           :: !(Proxy s)
  , _r1           :: !(Proxy s)
  , _replayMemory :: !(Maybe (ReplayMemory s))
  }
makeLenses ''Proxies

instance NFData s => NFData (Proxies s) where
  rnf (Proxies rhoMin rho psiV v w r0 r1 repMem) = rnf rhoMin `seq` rnf rho `seq` rnf psiV
    `seq` rnf v `seq` rnf w `seq` rnf r0 `seq` rnf r1 `seq` rnf repMem


allProxies :: Proxies s -> [Proxy s]
allProxies pxs = [pxs ^. rhoMinimum, pxs ^. rho, pxs ^. psiV, pxs ^. v, pxs ^. w, pxs ^. r0, pxs ^. r1]
