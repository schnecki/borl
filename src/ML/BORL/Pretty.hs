{-# LANGUAGE OverloadedStrings #-}

module ML.BORL.Pretty
    ( prettyTable
    , prettyBORL
    , prettyBORLTables
    ) where

import           ML.BORL.Parameters
import qualified ML.BORL.Proxy      as P
import           ML.BORL.Type

import           Control.Lens
import           Data.Function      (on)
import           Data.List          (find, sortBy)
import qualified Data.Map.Strict    as M
import           Prelude            hiding ((<>))
import           Text.PrettyPrint   as P
import           Text.Printf

commas :: Int
commas = 4

printFloat :: Double -> Doc
printFloat x = text $ printf ("%." ++ show commas ++ "f") x

-- prettyE :: (Ord s, Show s) => BORL s -> Doc
-- prettyE borl = case (borl^.r1,borl^.r0) of
--   (P.Table rm1, P.Table rm0) -> prettyTable id (P.Table $ M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0))
--   where subtr (k,v1) (_,v2) = (k,v1-v2)

prettyProxy :: (Ord k', Show k') => (k -> k') -> P.Proxy k -> Doc
prettyProxy = prettyTable

prettyTable :: (Ord k', Show k') => (k -> k') -> P.Proxy k -> Doc
prettyTable prettyKey p = vcat $ prettyTableRows prettyKey p

prettyTableRows :: (Ord k', Show k') => (k -> k') -> P.Proxy k -> [Doc]
prettyTableRows prettyAction p = case p of
  P.Table m -> map (\(k,val) -> text (show k) <> colon <+> printFloat val) (sortBy (compare `on` fst) $ M.toList (M.mapKeys prettyAction m))
  P.NN net conf -> [text "No tabular representation possible"]

-- prettyTablesStateAction :: (Ord k, Ord k1, Show k, Show k1) => P.Proxy k -> P.Proxy k1 -> Doc
-- prettyTablesStateAction m1 m2 = vcat $ zipWith (\x y -> x $$ nest 40 y) (prettyTableRows m1) (prettyTableRows m2)

prettyTablesState :: (Ord k', Ord k1', Show k', Show k1') => (k -> k') -> P.Proxy k -> (k1 -> k1') -> P.Proxy k1 -> Doc
prettyTablesState p1 m1 p2 m2 = vcat $ zipWith (\x y -> x $$ nest 40 y) (prettyTableRows p1 m1) (prettyTableRows p2 m2)


prettyBORLTables :: (Ord s, Show s) => Bool -> Bool -> Bool -> BORL s -> Doc
prettyBORLTables t1 t2 t3 borl =
  text "\n" $+$ text "Current state" <> colon $$ nest 45 (text (show $ borl ^. s)) $+$ text "Period" <> colon $$ nest 45 (integer $ borl ^. t) $+$ text "Alpha" <> colon $$
  nest 45 (printFloat $ borl ^. parameters . alpha) $+$
  text "Beta" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . beta) $+$
  text "Delta" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . delta) $+$
  text "Epsilon" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . epsilon) $+$
  text "Exploration" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . exploration) $+$
  text "Learning Random Actions" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . learnRandomAbove) $+$
  text "Gammas" <>
  colon $$
  nest 45 (text (show (printFloat $ borl ^. gammas . _1, printFloat $ borl ^. gammas . _2))) $+$
  text "Xi (ratio of W error forcing to V)" <>
  colon $$
  nest 45 (printFloat $ borl ^. parameters . xi) $+$
  text "Psi Rho/Psi V/Psi W" <>
  colon $$
  nest 45 (text (show (printFloat $ borl ^. psis . _1, printFloat $ borl ^. psis . _2, printFloat $ borl ^. psis . _3))) $+$
  (case borl ^. rho of
     Left val -> text "Rho" <> colon $$ nest 45 (printFloat val)
     Right m  -> text "Rho" $+$ prettyProxy prettyAction m) $$
  prBoolTblsStateAction t1 (text "V" $$ nest 40 (text "W")) (borl ^. v) (borl ^. w) $+$
  prBoolTblsState t2 (text "Psi V" $$ nest 40 (text "Psi W")) (borl ^. psiStates._2) (borl ^. psiStates._3) $$
  -- prBoolTbls t2 (text "V+Psi V" $$ nest 40 (text "W + Psi W")) (M.fromList $ zipWith add (M.toList $ borl ^. v) (M.toList $ borl ^. psiStates._2))
  -- (M.fromList $ zipWith add (M.toList $ borl ^. w) (M.toList $ borl ^. psiStates._3)) $+$
  prBoolTblsStateAction t2 (text "R0" $$ nest 40 (text "R1")) (borl ^. r0) (borl ^. r1) $+$
  (if t3 then text "E" $+$ prettyTable prettyAction e else empty) $+$
  text "Visits [%]" $+$ prettyTable id (P.Table vis)
  where
    e = case (borl^.r1,borl^.r0) of
      (P.Table rm1, P.Table rm0) -> P.Table $ M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0)
      (P.NN rm1 conf, P.NN rm0 _) -> undefined -- P.NN $ M.fromList $ zipWith subtr (M.toList rm1) (M.toList rm0)
    vis = M.map (\x -> 100 * fromIntegral x / fromIntegral (borl ^. t)) (borl ^. visits)
    subtr (k, v1) (_, v2) = (k, v1 - v2)
    add (k, v1) (_, v2) = (k, v1 + v2)
    prBoolTblsState True h m1 m2 = h $+$ prettyTablesState id m1 id m2
    prBoolTblsState False _ _ _  = empty
    prBoolTblsStateAction True h m1 m2 = h $+$ prettyTablesState prettyAction m1 prettyAction m2
    prBoolTblsStateAction False _ _ _  = empty
    prettyAction (s,aIdx) = (s, maybe "unkown" (actionName . snd) (find ((== aIdx) . fst) (borl ^. actionList)))


prettyBORL :: (Ord s, Show s) => BORL s -> Doc
prettyBORL = prettyBORLTables True True True


instance (Ord s, Show s) => Show (BORL s) where
  show borl = show (prettyBORL borl)
