package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
)

func forwardPass(network *[][][]float64, weights *[][]float64, input [inputSize]float64, output [outputSize]float64, batchIndex int) ([]float64, float64) {
	copy((*network)[0][batchIndex], input[:])

	var biasIndex int

	for i := 0; i < len(*weights)-1; i++ {
		for j := range (*network)[i+1][batchIndex] {
			sum := 0.0
			for k := range (*network)[i][batchIndex] {
				weightIndex := k*len((*network)[i+1][batchIndex]) + j
				sum += (*network)[i][batchIndex][k] * (*weights)[i][weightIndex]
			}

			if i < len(*weights)-2 {
				sum += (*weights)[len(*weights)-1][biasIndex]
				biasIndex++
			}

			(*network)[i+1][batchIndex][j] = sum
		}

		if i < len(*weights)-2 {
			normalize(&(*network)[i+1][batchIndex])
			activationFunction(&(*network)[i+1][batchIndex])
		}
	}

	normalize(&(*network)[len(*network)-1][batchIndex])

	finalOutput := (*network)[len(*network)-1][batchIndex]
	return finalOutput, getError(finalOutput, output)
}

func initWeights(weights *[][]float64, networkSize []uint8) {
	var numBaises int

	for i := range networkSize {
		if i > 0 {
			var size int = int(networkSize[i]) * int(networkSize[i-1])
			buffer := make([]float64, size)

			for j := 0; j < size; j++ {
				buffer[j] = rand.Float64()*2 - 1
				//rand.NormFloat64() * math.Sqrt(2.0/float64(networkSize[i]+networkSize[i-1])) may work better
			}

			*weights = append(*weights, buffer)

			if i < len(networkSize)-1 {
				numBaises += int(networkSize[i])
			}
		}
	}

	buffer := make([]float64, numBaises)

	for j := 0; j < numBaises; j++ {
		buffer[j] = rand.Float64()*2 - 1
	}

	*weights = append(*weights, buffer)
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

func activationFunction(array *[]float64) {
	for i, x := range *array {
		x *= 6
		(*array)[i] = (x / (1 + math.Abs(x)))
	}
}

func getError(output []float64, expected [outputSize]float64) float64 {
	var err float64

	for i := range output {
		err += math.Pow(expected[i]-output[i], 2)
	}

	err /= float64(outputSize)

	return err
}

func backwardPass(network *[][][]float64, weights *[][]float64, expectedOutputs [][outputSize]float64, batchSize int, learningRate float64, numGoroutines int) {
	numLayers := len(*network)

	// Prepare gradient accumulators
	gradWeights := make([][]float64, len(*weights))
	for i := range gradWeights {
		gradWeights[i] = make([]float64, len((*weights)[i]))
	}

	var mutex sync.Mutex // To safely update gradients

	// Calculate the size of each chunk for parallel processing
	chunkSize := (batchSize + numGoroutines - 1) / numGoroutines

	var wg sync.WaitGroup

	// Launch Go routines for each chunk
	for g := 0; g < numGoroutines; g++ {
		start := g * chunkSize
		end := min((g+1)*chunkSize, batchSize)

		if start >= end {
			break // No work for this Go routine
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			// Local gradient accumulator for this Go routine
			localGradWeights := make([][]float64, len(*weights))
			for i := range localGradWeights {
				localGradWeights[i] = make([]float64, len((*weights)[i]))
			}

			for batchIndex := start; batchIndex < end; batchIndex++ {
				expectedOutput := expectedOutputs[batchIndex]

				// Prepare deltas for each layer
				deltas := make([][]float64, numLayers)
				for i := range deltas {
					deltas[i] = make([]float64, len((*network)[i][batchIndex]))
				}

				// Calculate output layer deltas
				for i := 0; i < len((*network)[numLayers-1][batchIndex]); i++ {
					output := (*network)[numLayers-1][batchIndex][i]
					err := output - expectedOutput[i]
					deltas[numLayers-1][i] = 2 * err // Derivative of MSE
				}

				// Backpropagate deltas through hidden layers
				for l := numLayers - 2; l > 0; l-- {
					for i := 0; i < len((*network)[l][batchIndex]); i++ {
						sum := 0.0
						for j := 0; j < len((*network)[l+1][batchIndex]); j++ {
							weightIndex := i*len((*network)[l+1][batchIndex]) + j
							sum += (*weights)[l][weightIndex] * deltas[l+1][j]
						}
						activationDerivative := 1 / math.Pow(1+math.Abs((*network)[l][batchIndex][i]), 2) // Softsign derivative
						deltas[l][i] = sum * activationDerivative
					}
				}

				// Accumulate weight gradients for this chunk
				for l := 0; l < numLayers-1; l++ {
					for i := 0; i < len((*network)[l][batchIndex]); i++ {
						for j := 0; j < len((*network)[l+1][batchIndex]); j++ {
							weightGradient := deltas[l+1][j] * (*network)[l][batchIndex][i]
							weightIndex := i*len((*network)[l+1][batchIndex]) + j
							localGradWeights[l][weightIndex] += weightGradient
						}
					}
				}
			}

			// Safely merge local gradients into the global accumulator
			mutex.Lock()
			for l := 0; l < len(gradWeights); l++ {
				for i := range gradWeights[l] {
					gradWeights[l][i] += localGradWeights[l][i]
				}
			}
			mutex.Unlock()
		}(start, end)
	}

	// Wait for all Go routines to finish
	wg.Wait()

	// Apply accumulated gradients to weights
	for l := 0; l < len(*weights); l++ {
		for i := range (*weights)[l] {
			(*weights)[l][i] -= learningRate * gradWeights[l][i] / float64(batchSize) // Normalize by batch size
		}
	}
}

func saveModel(size []uint8, model [][]float64) error {
	// Open the file for writing. Create it if it doesn't exist.
	file, err := os.Create("model.gonet")
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	// Write the size to the file.
	metaStr := uint8ToString(size)
	_, err = file.WriteString(metaStr + "\n")
	if err != nil {
		return fmt.Errorf("failed to write size: %w", err)
	}

	// Write each row of the model to the file.
	for _, row := range model {
		// Convert each float64 to a string and join them with spaces.
		strRow := strings.TrimSpace(strings.Join(floatToStrings(row), " "))
		_, err := file.WriteString(strRow + "\n")
		if err != nil {
			return fmt.Errorf("failed to write model data: %w", err)
		}
	}

	return nil
}

// uint8ToString converts a []uint8 to a single space-separated string.
func uint8ToString(data []uint8) string {
	strs := make([]string, len(data))
	for i, v := range data {
		strs[i] = fmt.Sprintf("%d", v)
	}
	return strings.Join(strs, " ")
}

// floatToStrings converts a slice of float64 to a slice of strings.
func floatToStrings(row []float64) []string {
	strs := make([]string, len(row))
	for i, v := range row {
		strs[i] = fmt.Sprintf("%g", v)
	}
	return strs
}
