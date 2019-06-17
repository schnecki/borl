{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE TemplateHaskell #-}


module ML.BORL.NeuralNetwork.ReplayMemory where

import           ML.BORL.Types

import           Control.DeepSeq
import           Control.Lens
import qualified Data.Vector.Mutable as V
-- import qualified Data.Vector.Generic as V

import           System.Random


data ReplayMemory s = ReplayMemory
  { _replayMemoryVector :: V.IOVector (State s, ActionIndex, Bool, Double, StateNext s, EpisodeEnd)
  , _replayMemorySize   :: Int
  }
makeLenses ''ReplayMemory


instance NFData (ReplayMemory s) where
  rnf (ReplayMemory !_ s) = rnf s

-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: Period -> (State s, ActionIndex, Bool, Double, StateNext s, EpisodeEnd) -> ReplayMemory s -> IO (ReplayMemory s)
addToReplayMemory p e (ReplayMemory vec sz) = do
  let idx = p `mod` fromIntegral sz
  V.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoryElements :: Period -> Batchsize -> ReplayMemory s -> IO [(State s, ActionIndex, Bool, Double, StateNext s, EpisodeEnd)]
getRandomReplayMemoryElements t bs (ReplayMemory vec sz) = do
  let maxIdx = fromIntegral (min t (fromIntegral sz-1)) :: Int
  let len = min bs (maxIdx + 1)
  g <- newStdGen
  let rands = take len $ randomRs (0,maxIdx) g
  mapM (V.read vec) rands
