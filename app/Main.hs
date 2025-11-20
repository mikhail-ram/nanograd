module Main where

import Nanograd
import Data.List (foldl')
import System.Random (newStdGen, randomRs)

-- Network Architecture
-- 2 inputs -> 3 neurons (hidden) -> 1 neuron (output)
initialNetwork :: IO Network
initialNetwork = do
  gen <- newStdGen
  let randoms = randomRs (-1.0, 1.0) gen
      makeNeuron n ws = (Neuron (take n ws) (head $ drop n ws), drop (n + 1) ws)
      (n1, r1) = makeNeuron 2 randoms
      (n2, r2) = makeNeuron 2 r1
      (n3, r3) = makeNeuron 2 r2
      (n4, _) = makeNeuron 3 r3
  return [ DenseLayer [n1, n2, n3]
         , ActivationLayer Sigmoid
         , DenseLayer [n4]
         , ActivationLayer Sigmoid
         ]

-- XOR training data
xorInputs :: [[Double]]
xorInputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

xorTargets :: [[Double]]
xorTargets = [[0], [1], [1], [0]]

learningRate :: Double
learningRate = 0.1
epochs :: Int
epochs = 40000

trainStep :: Network -> InputVector -> OutputVector -> Network
trainStep net input target =
  let trace = forwardTrace net input
      predicted = last trace
      initialGradient = zipWith (-) predicted target
      (_, gradients) = backward net trace initialGradient
  in update net gradients learningRate

main :: IO ()
main = do
  net <- initialNetwork
  putStrLn "Initial Network:"
  print net
  let trainingData = cycle (zip xorInputs xorTargets)
  let trainedNet = foldl' (\n (input, target) -> trainStep n input target) net (take epochs trainingData)
  putStrLn "\nTrained Network:"
  print trainedNet
  putStrLn "\nTesting:"
  mapM_ (\(input, target) -> do
    let output = forward trainedNet input
    putStrLn $ "Input: " ++ show input ++ ", Target: " ++ show target ++ ", Predicted: " ++ show output
    ) (zip xorInputs xorTargets)
