# **GoNet: Lightweight Neural Network in Go** test

GoNet is a lightweight, minimalistic neural network implementation written in Go. Designed with simplicity in mind, it currently supports multilayer perceptrons (MLPs) and uses batch stochastic gradient descent with backpropagation for training.

---

## **Features**
- ✔️ **Core Functionality**: Implements basic MLP architecture.  
- ✔️ **Training Algorithm**: Batch stochastic gradient descent with backpropagation.  
- ✔️ **Lightweight Design**: Focused on simplicity and clarity, making it an excellent starting point for those interested in learning about neural networks.  

---

## **Current Limitations**
GoNet is still under development and lacks the following features:
- ❌ **CSV Data Reading**: Built-in support for loading data from CSV files.  
- ❌ **Model Execution Post-Training**: The ability to load a saved model and run it for inference.  

These features may be added in the future, but active development may be limited due to other commitments.

---

## **Usage**
To get started with GoNet, you can install the library using the following command:  
```bash
go get github.com/AdamJacuch/GoNet@latest
```

---

## **Sample Implementation**
Below is an example of how to use GoNet to train and test a simple neural network:

```go
package main

import (
	"fmt"

	gonet "github.com/AdamJacuch/GoNet"
)

func main() {
	gonet.NumEpochs = 1000
	gonet.BatchSize = 4

	data := [][]float64{
		{0, 0, 1, 1, -1, 1, 0, -1},
		{1, 1, 1, 1, -1, 1, 0, 1},
		{-1, 0, -1, 1, -1, -1, -1},
		{-1, 1, -1, 1, -1, 1, -1},
		{1, -1, 1, -1, 0, 0, 0, 0},
		{1, 1, 0, 0, 0, 0, 1, 1},
		{-1, -1, 0, 0, 0, 0, -1, -1},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{1, 1, 1, 1, 1, 1, 1, 1},
		{0, 0, 0, 0, 0, 0, 0, 0},
		{-1, -1, -1, -1, -1, -1, -1, -1},
		{1, 1, 0, 0, 0, 0, -1, -1},
	}
	outputs := [][]float64{
		{0, 1, 0},
		{0, 0, 1},
		{1, 0, 0},
		{0, 1, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{0, 1, 0},
		{1, 0, 0},
		{0, 1, 0},
	}

	nn := gonet.NewNet([]int{8, 5, 3})

	nn.Train(data, outputs, 0.01, 10)

	output := nn.ForwardPass([]float64{-1, -1, 1, 1, 1, 1, 1, 1}) // should be {0, 0, 1}
	gonet.Softmax(&output)
	fmt.Println(output)
}
```

---

## **Contributions**
Contributions are welcome!  
- Feel free to submit a **pull request**.  
- Open an **issue** to suggest new features or report bugs.  

---

## **Disclaimer**
This project is a work in progress and may not be actively maintained.  
**Use it as-is or as a foundation for your own projects.**

---
