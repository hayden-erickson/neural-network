package nn

import (
	"math/rand"

	"github.com/hayden-erickson/neural-network/la"
)

type SGD struct {
	Activation la.OP
	APrime     la.OP
	CostPrime  la.BOP
	Eta        float64
	Net        Network
}

func (sgd SGD) Run(trainingData []Example, epochs, miniBatchSize int) Network {

	N := len(trainingData)
	M := miniBatchSize

	var out Network
	CopyNetwork(&out, &sgd.Net)

	for e := 0; e < epochs; e++ {
		shuffled := shuffle(trainingData)

		// N must be a multiple of miniBatchSize
		for i := 0; i < (N / M); i++ {
			sgd.updateMiniBatch(shuffled[(i*M):((i+1)*M)], &out)
		}
	}

	return out
}

func (sgd SGD) MRun(trainingData []Example, epochs, miniBatchSize int) Network {
	var out Network
	CopyNetwork(&out, &sgd.Net)

	for i := 0; i < epochs; i++ {
		shuffled := shuffle(trainingData)

		for j := 0; j < len(trainingData)/miniBatchSize; j++ {
			inputs, desired := miniBatchToMatricies(shuffled[(j * miniBatchSize):((j + 1) * miniBatchSize)])
			deltaW, deltaB := sgd.Net.MBackProp(inputs, desired, sgd.Activation, sgd.APrime, sgd.CostPrime)
			for k, _ := range deltaW {
				out.Weights[k] = la.MSUM(sgd.Net.Weights[k], la.MSCALE(deltaW[k], -sgd.Eta))
				out.Biases[k] = la.VSUM(sgd.Net.Biases[k], la.VSCALE(deltaB[k], -sgd.Eta))
			}
		}
	}

	return out
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

func (sgd SGD) updateMiniBatch(miniBatch []Example, out *Network) {
	totalW := make([]la.Matrix, len(sgd.Net.Weights))
	totalB := make([][]float64, len(sgd.Net.Biases))

	for i := range sgd.Net.Weights {
		totalW[i] = la.ZeroMatrix(sgd.Net.Weights[i].Shape()[0], sgd.Net.Weights[i].Shape()[1])
		totalB[i] = make([]float64, len(sgd.Net.Biases[i]))
	}

	for _, e := range miniBatch {
		nablaW, nablaB := sgd.Net.BackProp(e, sgd.Activation, sgd.APrime, sgd.CostPrime)

		for i := range nablaW {
			totalW[i] = la.MSUM(totalW[i], nablaW[i])
			totalB[i] = la.VSUM(totalB[i], nablaB[i])
		}
	}

	lrOverBatch := -(sgd.Eta / float64(len(miniBatch)))

	for i := range sgd.Net.Weights {
		out.Weights[i] =
			la.MSUM(sgd.Net.Weights[i],
				la.MSCALE(totalW[i], lrOverBatch))

		out.Biases[i] =
			la.VSUM(sgd.Net.Biases[i],
				la.VSCALE(totalB[i], lrOverBatch))
	}
}
