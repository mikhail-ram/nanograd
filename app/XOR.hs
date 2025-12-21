module Main where

import Data.List (foldl')
import System.Random (newStdGen, randomRs)
import qualified Data.Massiv.Array as M
import Nanograd

-- Network Architecture
-- 2 inputs -> 3 neurons (hidden) -> 1 neuron (output)
initialNetwork :: IO Network
initialNetwork = do
  gen <- newStdGen
  let
    randoms = randomRs (-1.0, 1.0) gen
    (hiddenLayer, r1) = mkDenseLayer 2 3 randoms
    (outputLayer, _) = mkDenseLayer 3 1 r1
  return [ hiddenLayer
         , ActivationLayer Sigmoid
         , outputLayer
         , ActivationLayer Sigmoid
         ]

-- XOR training data
xorInputs :: [[Double]]
xorInputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

xorInputsTensor :: [InputVector]
xorInputsTensor = map (M.fromList M.Seq) xorInputs

xorTargets :: [[Double]]
xorTargets = [[0], [1], [1], [0]]

xorTargetsTensor :: [OutputVector]
xorTargetsTensor = map (M.fromList M.Seq) xorTargets

learningRate :: Double
learningRate = 0.1

epochs :: Int
epochs = 40000

main :: IO ()
main = do
  net <- initialNetwork
  putStrLn "Initial Network:"
  print net
  putStrLn "\nTraining Network..."
  let
    trainingData = cycle (zip xorInputsTensor xorTargetsTensor)
    trainingCycle = take epochs trainingData
    trainedNet = foldl' (\n (input, target) -> trainStep n input target learningRate) net trainingCycle
  putStrLn "Training complete."
  putStrLn "\nTrained Network:"
  print trainedNet
  putStrLn "\nTesting:"
  mapM_ (\(input, target) -> do
    let output = forward trainedNet input
    putStrLn $ "Input: " ++ show input ++ ", Target: " ++ show target ++ ", Predicted: " ++ show output
    ) (zip xorInputsTensor xorTargetsTensor)
