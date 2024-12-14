package gonet

import (
	"fmt"
	"math"
	"math/rand"
)

var (
	NumEpochs int
	BatchSize int
)

type Network struct {
	neurons [][]float64
	weights Weights

	gradients Weights
	numGrads  int
}

type Weights struct {
	connections [][]float64
	biases      [][]float64
}

func NewNet(size []int) Network {
	var net Network

	for i := 0; i < len(size)-1; i++ {
		weightSize := size[i] * size[i+1]
		buffer := make([]float64, weightSize)
		for j := 0; j < weightSize; j++ {
			buffer[j] = rand.Float64()*2 - 1
		}
		net.neurons = append(net.neurons, make([]float64, size[i]))
		net.weights.connections = append(net.weights.connections, buffer)
		if i > 0 {
			net.weights.biases = append(net.weights.biases, make([]float64, size[i]))
		}
	}

	net.neurons = append(net.neurons, make([]float64, size[len(size)-1]))

	return net
}

func (net *Network) Train(data [][]float64, outputs [][]float64, lr float64, numData int) {
	for i := 0; i < NumEpochs; i++ {
		for j := 0; j < len(outputs)/BatchSize; j++ {
			var err float64
			for k := 0; k < BatchSize; k++ {
				output := net.ForwardPass(data[(j*BatchSize)+k])
				net.backwardsPass(outputs[(j*BatchSize)+k], lr)
				err += calculateError(output, outputs[(j*BatchSize)+k])
			}
			net.update()
			err /= float64(BatchSize)

			if (i+1)%(NumEpochs/numData) == 0 {
				fmt.Printf("Epoch: %v, Batch %v, Error: %.5f\n", i+1, j+1, err)
			}
		}
		if len(outputs)%BatchSize > 0 {
			start := len(outputs) - (len(outputs) % BatchSize)
			if start > 0 {
				start--
			}
			var err float64
			for k := start; k < len(outputs); k++ {
				output := net.ForwardPass(data[k])
				net.backwardsPass(outputs[k], lr)
				err += calculateError(output, outputs[k])
			}
			net.update()
			err /= float64(len(outputs) % BatchSize)

			if (i+1)%(NumEpochs/numData) == 0 {
				fmt.Printf("Epoch: %v, Batch %v, Error: %.5f\n", i+1, len(outputs)/BatchSize+1, err)
			}
		}
	}
}

func (net *Network) ForwardPass(input []float64) []float64 {
	copy(net.neurons[0], input)

	for i := 0; i < len(net.neurons)-1; i++ {
		for j := 0; j < len(net.neurons[i]); j++ {
			for k := 0; k < len(net.neurons[i+1]); k++ {
				if j == 0 {
					net.neurons[i+1][k] = net.neurons[i][j] * net.weights.connections[i][(j*len(net.neurons[i+1]))+k]
				} else {
					net.neurons[i+1][k] += net.neurons[i][j] * net.weights.connections[i][(j*len(net.neurons[i+1]))+k]
				}
			}
		}

		if i < len(net.neurons)-2 {
			for k := 0; k < len(net.neurons[i+1]); k++ {
				net.neurons[i+1][k] += net.weights.biases[i][k]
			}
			normalize(&net.neurons[i+1])
			for k := 0; k < len(net.neurons[i+1]); k++ {
				net.neurons[i+1][k] *= 6
				net.neurons[i+1][k] = net.neurons[i+1][k] / (1 + math.Abs(net.neurons[i+1][k]))
			}
		}
	}

	return net.neurons[len(net.neurons)-1]
}

func (net *Network) backwardsPass(expected []float64, learningRate float64) {
	networkSize := len(net.neurons)

	output := net.neurons[networkSize-1]

	deltas := make([][]float64, networkSize-1)

	buffer := make([]float64, len(net.neurons[networkSize-1]))
	for i := 0; i < len(net.neurons[networkSize-1]); i++ {
		buffer[i] = 2 * (output[i] - expected[i])
	}
	deltas[networkSize-2] = buffer

	for i := networkSize - 3; i >= 0; i-- {
		layerDelta := make([]float64, len(net.neurons[i+1]))
		for j := 0; j < len(net.neurons[i+1]); j++ {
			var deltaSum float64
			for k := 0; k < len(net.neurons[i+2]); k++ {
				deltaSum += deltas[i+1][k] * net.weights.connections[i+1][j*len(net.neurons[i+2])+k]
			}
			derivative := 6 / math.Pow(1+math.Abs(net.neurons[i+1][j]), 2)
			layerDelta[j] = deltaSum * derivative
		}
		deltas[i] = layerDelta
	}

	for i := 0; i < len(net.weights.connections); i++ {
		for j := 0; j < len(net.neurons[i]); j++ {
			for k := 0; k < len(net.neurons[i+1]); k++ {
				gradientChange := learningRate * deltas[i][k] * net.neurons[i][j]
				net.gradients.connections[i][j*len(net.neurons[i+1])+k] += gradientChange
			}
		}

		if i < len(net.weights.biases) {
			for k := 0; k < len(net.neurons[i+1]); k++ {
				biasChange := learningRate * deltas[i][k]
				net.gradients.biases[i][k] += biasChange
			}
		}
	}

	net.numGrads++
}

func (net *Network) update() {
	for i := range net.weights.connections {
		for j := range net.weights.connections[i] {
			net.weights.connections[i][j] -= net.gradients.connections[i][j]
		}
	}
	for i := range net.weights.biases {
		for j := range net.weights.biases[i] {
			net.weights.biases[i][j] -= net.gradients.biases[i][j]
		}
	}

	net.gradients = Weights{}
}

func normalize(array *[]float64) {
	var max float64 = 0.01

	for i := range *array {
		if math.Abs((*array)[i]) > max {
			max = math.Abs((*array)[i])
		}
	}

	for i := range *array {
		(*array)[i] /= max
	}
}

func Softmax(array *[]float64) {
	var min float64 = 99
	var max float64 = -99

	for i := range *array {
		if (*array)[i] < min {
			min = (*array)[i]
		}
		if (*array)[i] > max {
			max = (*array)[i]
		}
	}

	for i := range *array {
		(*array)[i] -= min
		(*array)[i] /= (max - min)
	}
}

func calculateError(output []float64, expected []float64) float64 {
	var err float64 = 0

	for i := range output {
		err += math.Pow(expected[i]-output[i], 2)
	}

	return err / float64(len(output))
}
