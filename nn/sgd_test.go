package nn_test

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/hayden-erickson/neural-network/la"
	. "github.com/hayden-erickson/neural-network/nn"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// an example network to convert decimal number to binary
// 0 <= num < 16
type testEx struct {
	num int
}

func (te testEx) GetInput() []float64 {
	out := make([]float64, 16)
	out[te.num] = 1
	return out
}

func (te testEx) GetOutput() []float64 {
	out := make([]float64, 4)

	for i, _ := range out {
		out[i] = float64((te.num & (1 << uint(3-i))) >> uint(3-i))
	}

	return out
}

func generateExamples(n int) []Example {
	out := make([]Example, n)

	for i, _ := range out {
		out[i] = testEx{rand.Intn(16)}
	}

	return out
}

var _ = Describe("Sgd", func() {
	var new, original Network
	var examples []Example

	BeforeEach(func() {
		original, _ = NewNetwork([]int{16, 4})

		sgd := SGD{
			Activation: sigmoid,
			APrime:     sigmoidPrime,
			CostPrime:  costPrime,
			Eta:        5,
			Net:        original,
		}

		rand.Seed(time.Now().UnixNano())

		examples = generateExamples((rand.Intn(50) * 10) + 1000)

		new = sgd.Run(examples, 50, 100)
	})

	It("returns a network with a lower cost than the original", func() {
		var newDesired, newActual, origDesired, origActual [][]float64

		for _, e := range examples {
			newActual = append(newActual, new.Prop(e.GetInput(), sigmoid))
			newDesired = append(newDesired, e.GetOutput())
			origActual = append(origActual, original.Prop(e.GetInput(), sigmoid))
			origDesired = append(origDesired, e.GetOutput())
		}

		newCost := cost(newDesired, newActual)
		origCost := cost(origDesired, origActual)

		fmt.Printf("orig: %f, new: %f", origCost, newCost)

		Expect(newCost).To(BeNumerically(`<`, origCost))
	})

	XIt("returns a network with higher accuracy than original", func() {
		N := 100
		testData := generateExamples(N)
		newCorrect := 0
		origCorrect := 0

		for _, e := range testData {
			// fmt.Println(`=== num ===`)
			if matchesDesired(new.Prop(e.GetInput(), sigmoid), e.GetOutput()) {
				newCorrect++
			}

			if matchesDesired(original.Prop(e.GetInput(), sigmoid), e.GetOutput()) {
				origCorrect++
			}
		}

		Expect(newCorrect / N).To(BeNumerically(`>`, origCorrect/N))

	})
})

func matchesDesired(actual, desired []float64) bool {
	matches := true

	for i, _ := range actual {
		// fmt.Println(desired[i])
		// fmt.Println(actual[i])
		if desired[i] == 1.0 && actual[i] < 0.5 {
			return false
		}

		if desired[i] == 0.0 && actual[i] >= 0.5 {
			return false
		}
	}

	return matches
}

func sigmoid(z []float64) []float64 {
	act := func(z float64) float64 {
		return 1 / (1 + math.Exp(-z))
	}

	return la.Map(z, act)
}

func sigmoidPrime(z []float64) []float64 {
	oneMinus := func(x float64) float64 {
		return 1 - x
	}

	return la.MULT(sigmoid(z), la.Map(sigmoid(z), oneMinus))
}

func cost(desired, actual [][]float64) float64 {

	intermediate := make([]float64, len(desired))

	for i, _ := range desired {
		yMinA := la.SUB(desired[i], actual[i])
		intermediate[i] = la.AddReduce(la.MULT(yMinA, yMinA))
	}

	return la.AddReduce(intermediate) / float64(2*len(desired))
}

func costPrime(actual, desired []float64) []float64 {
	return la.SUB(actual, desired)
}
