package nn

import (
	"math/rand"

	"github.com/hayden-erickson/neural-network/la"
)

type SGD struct {
	Activation Differentiable
	Cost       Differentiable
	Eta        float64
	Net        Network
}

func (sgd SGD) MRun(trainingData []Example, epochs, miniBatchSize int) {
	shuffled := shuffle(trainingData)
	for i := 0; i < epochs; i++ {
		for j := 0; j < len(trainingData)/miniBatchSize; j++ {
			inputs, desired := miniBatchToMatricies(shuffled[(j * miniBatchSize):((j + 1) * miniBatchSize)])
			deltaW, deltaB := sgd.Net.MBackProp(inputs, desired, sgd.Activation, sgd.Cost)
			for k, _ := range deltaW {
				sgd.Net.Weights[k] = la.MSUM(sgd.Net.Weights[k], la.MSCALE(deltaW[k], -sgd.Eta))
				sgd.Net.Biases[k] = la.VSUM(sgd.Net.Biases[k], la.VSCALE(deltaB[k], -sgd.Eta))
			}
		}
	}
}

func (sgd SGD) Run(trainingData []Example, epochs, miniBatchSize int) {

	N := len(trainingData)
	M := miniBatchSize

	shuffled := shuffle(trainingData)

	totalW := make([]la.Matrix, len(sgd.Net.Weights))
	totalB := make([][]float64, len(sgd.Net.Biases))

	for i := range sgd.Net.Weights {
		totalW[i] = la.ZeroMatrix(sgd.Net.Weights[i].Shape()[0], sgd.Net.Weights[i].Shape()[1])
		totalB[i] = make([]float64, len(sgd.Net.Biases[i]))
	}

	for e := 0; e < epochs; e++ {
		// N must be a multiple of miniBatchSize
		for i := 0; i < (N / M); i++ {
			resetWeightsAndBiases(&totalW, &totalB)
			sgd.updateMiniBatch(shuffled[(i*M):((i+1)*M)], totalW, totalB)
		}
	}
}

func resetWeightsAndBiases(ws *[]la.Matrix, bs *[][]float64) {
	weights := *ws
	biases := *bs
	for i := 0; i < len(weights); i++ {
		resetMatrix(&weights[i])
		resetVector(&biases[i])
	}
}

func resetVector(v *[]float64) {
	vec := *v
	for i := 0; i < len(vec); i++ {
		vec[i] = 0
	}
}

func resetMatrix(m *la.Matrix) {
	mat := *m

	for i := 0; i < mat.Shape()[0]; i++ {
		for j := 0; j < mat.Shape()[1]; j++ {
			*mat.At(i, j) = 0
		}
	}
}

func miniBatchToMatricies(exs []Example) (input, desired la.Matrix) {
	N := len(exs)
	inputSize := len(exs[0].GetInput())
	outputSize := len(exs[0].GetOutput())
	input = la.ZeroMatrix(inputSize, N)
	desired = la.ZeroMatrix(outputSize, N)

	for j := 0; j < N; j++ {
		inputV := exs[j].GetInput()
		outputV := exs[j].GetOutput()

		for i := 0; i < inputSize; i++ {
			*input.At(i, j) = inputV[i]
		}

		for i := 0; i < outputSize; i++ {
			*desired.At(i, j) = outputV[i]
		}
	}

	return input, desired
}

func shuffle(a []Example) []Example {
	for i := len(a) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}

	return a
}

func (sgd SGD) updateMiniBatch(miniBatch []Example, totalW []la.Matrix, totalB [][]float64) {

	for _, e := range miniBatch {
		nablaW, nablaB := sgd.Net.BackProp(e, sgd.Activation, sgd.Cost)

		for i := range nablaW {
			totalW[i] = la.MSUM(totalW[i], nablaW[i])
			totalB[i] = la.VSUM(totalB[i], nablaB[i])
		}
	}

	lrOverBatch := -(sgd.Eta / float64(len(miniBatch)))

	for i := range sgd.Net.Weights {
		sgd.Net.Weights[i] =
			la.MSUM(sgd.Net.Weights[i],
				la.MSCALE(totalW[i], lrOverBatch))

		sgd.Net.Biases[i] =
			la.VSUM(sgd.Net.Biases[i],
				la.VSCALE(totalB[i], lrOverBatch))
	}
}
