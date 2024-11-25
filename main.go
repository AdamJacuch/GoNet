package main

import (
	"fmt"
)

const (
	inputSize  uint8 = 16
	outputSize uint8 = 4

	numInputs uint16 = 8
	batchSize int    = 4
	numEpochs int    = 1000

	numGoroutines int = 5
)

func main() {

	var network [][][]float64

	networkSize := []uint8{inputSize, 8, 6, outputSize}

	// Initialize the network with an additional dimension for batches
	for _, layerSize := range networkSize {
		batchLayer := make([][]float64, batchSize)
		for i := range batchLayer {
			batchLayer[i] = make([]float64, layerSize)
		}
		network = append(network, batchLayer)
	}

	var weights [][]float64

	initWeights(&weights, networkSize)

	input := [numInputs][inputSize]float64{
		{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
		{0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1},
		{0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 1, 1, 1, 1, 1, 1, 1, 1},
		{0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5},
		{0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
		{0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 0, 0, 0, 0, 0, 0, 0, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5},
		{0, 0, 0, 0, 0, 0, 0, 0, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5},
	}
	expectedOutput := [numInputs][outputSize]float64{
		{1, 0, 0, 0},
		{0, 0, 0, 1},
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 1, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 0, 0},
	}

	for epoch := 0; epoch < numEpochs; epoch++ {
		for start := 0; start < int(numInputs); start += int(batchSize) {
			end := min(start+int(batchSize), int(numInputs))

			// Extract the current batch
			currentInput := input[start:end]
			currentOutput := expectedOutput[start:end]

			// Perform forward pass for the batch
			var batchError float64
			for i := 0; i < end-start; i++ {
				_, err := forwardPass(&network, &weights, currentInput[i], currentOutput[i], i)
				batchError += err
			}

			// Perform backward pass
			backwardPass(&network, &weights, currentOutput[:end-start], end-start, 0.005, numGoroutines)

			// Print batch error
			if ((epoch + 1) % (numEpochs / 10)) == 0 {
				fmt.Printf("Epoch %d, Batch %d-%d: Error = %.6f\n", epoch+1, start, end, batchError/float64(end-start))
			}
		}
	}

	// save the model
	err := saveModel(networkSize, weights)

	if err != nil {
		panic(err)
	}
}
