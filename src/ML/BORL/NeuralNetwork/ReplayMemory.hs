{-# LANGUAGE BangPatterns    #-}
{-# LANGUAGE DeriveAnyClass  #-}
{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE Rank2Types      #-}
{-# LANGUAGE RankNTypes      #-}
{-# LANGUAGE TemplateHaskell #-}


module ML.BORL.NeuralNetwork.ReplayMemory where


import           Control.DeepSeq
import           Control.Lens
import           Control.Monad        (foldM)
import           Data.List            (genericLength)
import           Data.Maybe           (fromMaybe)
import qualified Data.Vector.Mutable  as VM
import qualified Data.Vector.Storable as V
import           GHC.Generics
import           System.Random

import           ML.BORL.Types

import           Debug.Trace

------------------------------ Replay Memories ------------------------------

type NStepCache = [Experience]

data ReplayMemories
  = ReplayMemoriesUnified !ReplayMemory                 -- ^ All experiences are saved in a single replay memory.
  | ReplayMemoriesPerActions ![ReplayMemory] NStepCache -- ^ Split replay memory size among different actions and choose bachsize uniformly among all sets of experiences.
  deriving (Generic)

replayMemories :: ActionIndex -> Lens' ReplayMemories ReplayMemory
replayMemories _ f (ReplayMemoriesUnified rm) = ReplayMemoriesUnified <$> f rm
replayMemories idx f (ReplayMemoriesPerActions rs cache) = (\x -> ReplayMemoriesPerActions (take idx rs ++ x : drop (idx+1) rs) cache) <$> f (rs !! idx)


instance NFData ReplayMemories where
  rnf (ReplayMemoriesUnified rm)       = rnf rm
  rnf (ReplayMemoriesPerActions rs ch) = rnf rs `seq` rnf ch

------------------------------ Replay Memory ------------------------------

type Experience = ((StateFeatures, V.Vector ActionIndex), ActionIndex, IsRandomAction, RewardValue, (StateNextFeatures, V.Vector ActionIndex), EpisodeEnd)


data ReplayMemory = ReplayMemory
  { _replayMemoryVector :: !(VM.IOVector Experience)
  , _replayMemorySize   :: !Int  -- size
  , _replayMemoryIdx    :: !Int  -- index to use when adding the next element
  , _replayMemoryMaxIdx :: !Int  -- in {0,..,size-1}
  }
makeLenses ''ReplayMemory

instance NFData ReplayMemory where
  rnf (ReplayMemory !_ s idx mx) = rnf s `seq` rnf idx `seq` rnf mx

addToReplayMemories :: Maybe Int -> Experience -> ReplayMemories -> IO ReplayMemories
addToReplayMemories _ e (ReplayMemoriesUnified rm) = ReplayMemoriesUnified <$> addToReplayMemory e rm
addToReplayMemories mNStep e@(_, idx, _, _, _, _) (ReplayMemoriesPerActions rs cache) =
  case mNStep of
    Nothing -> do
      let r = rs !! idx
      r' <- addToReplayMemory e r
      return $ ReplayMemoriesPerActions (take idx rs ++ r' : drop (idx + 1) rs) cache
    Just n
      | n - 1 == length cache -> do
        let r = rs !! idx -- last action decides
        r' <- foldM (flip addToReplayMemory) r cache'
        return $ ReplayMemoriesPerActions (take idx rs ++ r' : drop (idx + 1) rs) []
      | otherwise -> return $ ReplayMemoriesPerActions rs cache'
  where
    cache' = cache ++ [e]

-- | Add an element to the replay memory. Replaces the oldest elements once the predefined replay memory size is
-- reached.
addToReplayMemory :: Experience -> ReplayMemory -> IO ReplayMemory
addToReplayMemory e (ReplayMemory vec sz idx maxIdx) = do
  VM.write vec (fromIntegral idx) e
  return $ ReplayMemory vec sz ((idx+1) `mod` fromIntegral sz) (min (maxIdx+1) (sz-1))

-- | Get a list of random input-output tuples from the replay memory.
getRandomReplayMemoriesElements :: Maybe Int -> Batchsize -> ReplayMemories -> IO [[Experience]]
getRandomReplayMemoriesElements nStep bs (ReplayMemoriesUnified rm) = getRandomReplayMemoryElements nStep bs rm
getRandomReplayMemoriesElements nStep bs (ReplayMemoriesPerActions rs _) = concat <$> mapM (getRandomReplayMemoryElements nStep nr) rs
  where nr = ceiling (fromIntegral bs / genericLength rs :: Float)

-- | Get a list of random sequences of input-output tuples from the replay memory.
getRandomReplayMemoryElements :: Maybe Int -> Batchsize -> ReplayMemory -> IO [[Experience]]
getRandomReplayMemoryElements Nothing bs (ReplayMemory vec _ _ maxIdx) = do
  let len = min bs maxIdx
  g <- newStdGen
  let rands = take len $ randomRs (0, maxIdx) g
  mapM (fmap return . VM.read vec) rands
getRandomReplayMemoryElements (Just nStep) bs (ReplayMemory vec _ nextIdx maxIdx)
  | maxIdx <= 0 || maxIdx `mod` nStep /= 0 = return []
  | otherwise = do
    let len = min bs maxIdx
    g <- newStdGen
    let rands = take len $ randomRs (0, maxIdx `div` nStep) g
        mkRange r
          | nStep * r < nextIdx - 1 && nStep * r + nStep >= nextIdx = [nStep * r .. nextIdx - 1]
          | otherwise = [nStep * r .. nStep * r + nStep - 1]
        ranges = map mkRange rands
    mapM (mapM (VM.read vec)) ranges


-- | Size of replay memory (combined if it is a per action replay memory).
replayMemoriesSize :: ReplayMemories -> Int
replayMemoriesSize (ReplayMemoriesUnified m)       = m ^. replayMemorySize
replayMemoriesSize (ReplayMemoriesPerActions ms _) = sum $ map (view replayMemorySize) ms

replayMemoriesSubSize :: ReplayMemories -> Int
replayMemoriesSubSize (ReplayMemoriesUnified m)          = m ^. replayMemorySize
replayMemoriesSubSize (ReplayMemoriesPerActions (m:_) _) = m ^. replayMemorySize
replayMemoriesSubSize (ReplayMemoriesPerActions [] _)    = 0
