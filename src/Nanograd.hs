module Nanograd
  ( Neuron(..),
    ActivationFunc(..),
    Layer(..),
    Network,
    InputVector,
    OutputVector,
    forward,
    forwardTrace,
    showNetwork,
    backward,
    update,
    mkNeuron,
    mkDenseLayer,
    trainStep
  ) where

import Data.List (intercalate, scanl', transpose, mapAccumR)
import qualified Data.Massiv.Array as M

data Neuron = Neuron { weights :: Tensor1D
                     , bias    :: Double
                     } deriving Show

data ActivationFunc = Sigmoid deriving Show

data Layer =
    DenseLayer [Neuron]
  | ActivationLayer ActivationFunc
  deriving Show

data Activation = Activation
  { forwardPass  :: Double -> Double
  , backwardPass :: Double -> Double
  }

getActivation :: ActivationFunc -> Activation
getActivation Sigmoid = sigmoid

type Network = [Layer]

type Tensor1D = M.Array M.U M.Ix1 Double
type InputVector = Tensor1D
type OutputVector = Tensor1D

showNetwork :: Network -> String
showNetwork network = "[\n" ++ intercalate ",\n" (map show network) ++ "\n]"

sigmoid :: Activation
sigmoid = Activation
  { forwardPass = sigmoidFunc
  , backwardPass = \x ->
      let s = sigmoidFunc x
      in s * (1 - s)
  }
  where sigmoidFunc x = 1 / (1 + exp (-x))

forwardDense :: [Neuron] -> InputVector -> OutputVector
forwardDense layer input = M.fromList M.Seq (map forwardNeuron layer)
  where forwardNeuron neuron = M.sum (M.zipWith (*) input (weights neuron)) + (bias neuron)

forwardActivation :: Activation -> InputVector -> OutputVector
forwardActivation activation input = M.compute $ M.map (forwardPass activation) input

forwardLayer :: Layer -> InputVector -> OutputVector
forwardLayer (DenseLayer neurons) = forwardDense neurons
forwardLayer (ActivationLayer activationFunc) = forwardActivation (getActivation activationFunc)

forwardTrace :: Network -> InputVector -> [OutputVector]
forwardTrace network input = scanl' (flip forwardLayer) input network

forward :: Network -> InputVector -> OutputVector
forward network = last . forwardTrace network

type UpstreamGradient = Tensor1D
type DownstreamGradient = Tensor1D
type LayerGradient = [Neuron]

backwardActivation :: Activation -> InputVector -> UpstreamGradient -> DownstreamGradient
backwardActivation activation input upstream = M.compute $ M.zipWith (*) upstream derivatives :: Tensor1D
  where derivatives = M.compute $ M.map (backwardPass activation) input :: Tensor1D

backwardDense :: [Neuron] -> InputVector -> UpstreamGradient -> (DownstreamGradient, LayerGradient)
backwardDense neurons input upstream =
  let
    gradForNeuron up_err _neuron =
      let
        w_grad = M.compute $ M.map (* up_err) input
        b_grad = up_err
      in Neuron w_grad b_grad
    layerGrads = zipWith (gradForNeuron) (M.toList upstream) neurons
    weightsMatrix = map (M.toList . weights) neurons -- [[current 1 to prev1, current 1 to prev2], neuron 2 weights, ...]
    transposedWeights = transpose weightsMatrix -- [[current 1 to prev 1, current 2 to prev 1, ...], [current n to prev 2],...]
    downstream = M.fromList M.Seq $ map (\prev_node_weights -> let zipped = (M.compute $ M.zipWith (*) upstream ((M.fromList M.Seq prev_node_weights) :: Tensor1D)) :: Tensor1D in M.sum zipped) transposedWeights :: Tensor1D -- sum of multiplication of each output weight of previous single node to error caused in each layer node
  in (downstream, layerGrads)

backwardLayer :: Layer -> InputVector -> UpstreamGradient -> (DownstreamGradient, Maybe LayerGradient)
backwardLayer (DenseLayer neurons) input upstream = (downstream, Just layerGrad)
  where (downstream, layerGrad) = backwardDense neurons input upstream
backwardLayer (ActivationLayer activationFunc) input upstream = (downstream, Nothing)
  where activation = getActivation activationFunc
        downstream = backwardActivation activation input upstream

backward :: Network -> [OutputVector] -> UpstreamGradient -> (DownstreamGradient, [Maybe LayerGradient])
backward network trace initialGradient =
    let layerInputs = init trace
        layersWithInputs = zip network layerInputs
    in mapAccumR backwardStep initialGradient layersWithInputs
  where backwardStep :: UpstreamGradient -> (Layer, InputVector) -> (DownstreamGradient, Maybe LayerGradient)
        backwardStep up_grad (layer, input) = backwardLayer layer input up_grad

update :: Network -> [Maybe LayerGradient] -> Double -> Network
update network gradients learningRate = zipWith updateLayerGradient network gradients
  where updateLayerGradient :: Layer -> Maybe LayerGradient -> Layer
        updateLayerGradient (DenseLayer neurons) (Just grad_neurons) = DenseLayer (zipWith (updateNeuron) neurons grad_neurons)
        updateLayerGradient layer@(ActivationLayer _) (Just _) = layer
        updateLayerGradient layer Nothing = layer
        updateNeuron :: Neuron -> Neuron -> Neuron
        updateNeuron neuron grad_neuron = Neuron newWeights newBias
          where newWeights = M.compute $ M.zipWith (\w dw -> w - learningRate * dw) (weights neuron) (weights grad_neuron)
                newBias = bias neuron - learningRate * (bias grad_neuron)

mkNeuron :: Int -> [Double] -> (Neuron, [Double])
mkNeuron numInputs randoms = (neuron, remainingRandoms)
  where
    (neuronRandoms, remainingRandoms) = splitAt (numInputs + 1) randoms
    neuronWeights = M.fromList M.Seq (take numInputs neuronRandoms)
    neuronBias = head (drop numInputs neuronRandoms)
    neuron = Neuron { weights = neuronWeights, bias = neuronBias }

mkDenseLayer :: Int -> Int -> [Double] -> (Layer, [Double])
mkDenseLayer numInputsPerNeuron numNeurons randoms = (DenseLayer neurons, finalRandoms)
  where
    buildNeurons :: Int -> [Double] -> ([Neuron], [Double])
    buildNeurons 0 rands = ([], rands)
    buildNeurons n rands = 
      let
        (neuron, rands') = mkNeuron numInputsPerNeuron rands
        (restOfNeurons, rands'') = buildNeurons (n - 1) rands'
      in (neuron : restOfNeurons, rands'')
    (neurons, finalRandoms) = buildNeurons numNeurons randoms


trainStep :: Network -> InputVector -> OutputVector -> Double -> Network
trainStep net input target lr =
  let trace = forwardTrace net input
      predicted = last trace
      initialGradient = M.compute $ M.zipWith (-) predicted target
      (_, gradients) = backward net trace initialGradient
  in update net gradients lr
