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
    update
  ) where

import Data.List (intercalate, scanl', transpose, mapAccumR)

data Neuron = Neuron { weights :: [Double]
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

type Vector = [Double]
type InputVector = Vector
type OutputVector = Vector

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
forwardDense layer input = map forwardNeuron layer
  where forwardNeuron neuron = sum (zipWith (*) input (weights neuron)) + (bias neuron)

forwardActivation :: Activation -> InputVector -> OutputVector
forwardActivation activation = map (forwardPass activation)

forwardLayer :: Layer -> InputVector -> OutputVector
forwardLayer (DenseLayer neurons) = forwardDense neurons
forwardLayer (ActivationLayer activationFunc) = forwardActivation (getActivation activationFunc)

forwardTrace :: Network -> InputVector -> [OutputVector]
forwardTrace network input = scanl' (flip forwardLayer) input network

forward :: Network -> InputVector -> OutputVector
forward network = last . forwardTrace network

type UpstreamGradient = [Double]
type DownstreamGradient = [Double]
type LayerGradient = [Neuron]

backwardActivation :: Activation -> InputVector -> UpstreamGradient -> DownstreamGradient
backwardActivation activation input upstream = zipWith (*) upstream derivatives
  where derivatives = map (backwardPass activation) input

backwardDense :: [Neuron] -> InputVector -> UpstreamGradient -> (DownstreamGradient, LayerGradient)
backwardDense neurons input upstream =
  let gradForNeuron up_err neuron =
        let w_grad = map (* up_err) input
            b_grad = up_err
        in Neuron w_grad b_grad
      layerGrads = zipWith (gradForNeuron) upstream neurons
      weightsMatrix = map weights neurons -- [[current 1 to prev1, current 1 to prev2], neuron 2 weights, ...]
      transposedWeights = transpose weightsMatrix -- [[current 1 to prev 1, current 2 to prev 1, ...], [current n to prev 2],...]
      downstream = map (\prev_node_weights -> sum $ zipWith (*) upstream prev_node_weights) transposedWeights -- sum of multiplication of each output weight of previous single node to error caused in each layer node
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
        updateLayerGradient layer Nothing = layer
        updateNeuron :: Neuron -> Neuron -> Neuron
        updateNeuron neuron grad_neuron = Neuron newWeights newBias
          where newWeights = zipWith (\w dw -> w - learningRate * dw) (weights neuron) (weights grad_neuron)
                newBias = bias neuron - learningRate * (bias grad_neuron)
