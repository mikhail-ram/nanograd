{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric     #-}
-- TODO: add ReLU and Softmax activations
-- TODO: actual batching
-- TODO: real train-test-validation split

module Main where

import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import Data.Csv
import Data.List (foldl', elemIndex)
import GHC.Generics (Generic)
import System.Random (newStdGen, randomRs)
import Nanograd

data IrisRecord = IrisRecord
  { sepalLength :: Double
  , sepalWidth  :: Double
  , petalLength :: Double
  , petalWidth  :: Double
  , irisClass   :: String
  } deriving (Show, Generic)

instance FromRecord IrisRecord

irisClassToOneHot :: String -> OutputVector
irisClassToOneHot "Iris-setosa"     = [1.0, 0.0, 0.0]
irisClassToOneHot "Iris-versicolor" = [0.0, 1.0, 0.0]
irisClassToOneHot "Iris-virginica"  = [0.0, 0.0, 1.0]
irisClassToOneHot _                 = error "Unknown Iris class"

irisRecordToVectors :: IrisRecord -> (InputVector, OutputVector)
irisRecordToVectors r =
  ( [ sepalLength r
    , sepalWidth r
    , petalLength r
    , petalWidth r
    ]
  , irisClassToOneHot (irisClass r)
  )

-- Network Architecture
-- 4 inputs -> 5 neurons (hidden) -> 3 neurons (output)
initialNetwork :: IO Network
initialNetwork = do
  gen <- newStdGen
  let
    randoms = randomRs (-1.0, 1.0) gen
    (hiddenLayer, r1) = mkDenseLayer 4 5 randoms
    (outputLayer, _) = mkDenseLayer 5 3 r1
  return [ hiddenLayer
         , ActivationLayer Sigmoid
         , outputLayer
         , ActivationLayer Sigmoid
         ]

learningRate :: Double
learningRate = 0.01

epochs :: Int
epochs = 10000

indexOfMax :: [Double] -> Maybe Int
indexOfMax xs = elemIndex (maximum xs) xs

main :: IO ()
main = do
  csvData <- BL.readFile "data/iris/iris.data"
  let
    result :: Either String (V.Vector IrisRecord)
    result = decode NoHeader csvData
  case result of
    Left err -> putStrLn err
    Right v -> do
      putStrLn "Successfully parsed CSV:"
      V.forM_ v $ \r -> print r
      putStrLn "\nTransformed data:"
      let transformedData = V.map irisRecordToVectors v
      V.forM_ (V.take 5 (transformedData)) $ \(input, output) -> putStrLn $ "Input: " ++ show input ++ ", Output: " ++ show output
      putStrLn $ "\nTotal records: " ++ show (V.length transformedData)
      net <- initialNetwork
      putStrLn "\nInitial Network:"
      print net
      putStrLn "\nTraining Network..."
      let
        trainingData = V.toList (transformedData)
        trainingCycle = take epochs (cycle trainingData)
        trainedNet = foldl' (\n (input, target) -> trainStep n input target learningRate) net trainingCycle
      putStrLn "\nTraining complete."
      putStrLn "\nTrained Network:"
      print trainedNet
      putStrLn "\nEvaluating Network..."
      let testData = V.toList transformedData
      let
        results = map (\(input, target) ->
            let
              predictedRaw = forward trainedNet input
              predictedClass = indexOfMax predictedRaw 
              actualClass = indexOfMax target
            in (predictedClass, actualClass)
          ) testData
        correctPredictions = length [() | (Just p, Just a) <- results, p == a] 
        totalPredictions = length testData
        accuracy :: Double
        accuracy = (fromIntegral correctPredictions / fromIntegral totalPredictions) * 100
      putStrLn $ "Correctly predicted " ++ show correctPredictions ++ " out of " ++ show totalPredictions
      putStrLn $ "Accuracy: " ++ show accuracy ++ "%"
