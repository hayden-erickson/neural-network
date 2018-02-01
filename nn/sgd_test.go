package nn_test

import (
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
	var sgd SGD

	BeforeEach(func() {
		original, _ = NewNetwork([]int{16, 4})

		sgd = SGD{
			Activation: sigmoid,
			APrime:     sigmoidPrime,
			CostPrime:  costPrime,
			Eta:        5,
			Net:        original,
		}

		rand.Seed(time.Now().UnixNano())

		examples = generateExamples((rand.Intn(50) * 10) + 1000)

	})

	Describe("#MRun", func() {
		BeforeEach(func() {
			new = sgd.MRun(examples, 50, 100)
		})

		It("returns a network with lower cost than the original", func() {
			newCost, origCost := evaluateNets(examples, sigmoid, new, original)
			Expect(newCost).To(BeNumerically(`<`, origCost))
		})
	})

	Describe("#Run", func() {
		BeforeEach(func() {
			new = sgd.Run(examples, 50, 100)
		})

		It("returns a network with a lower cost than the original", func() {
			newCost, origCost := evaluateNets(examples, sigmoid, new, original)
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
})

func evaluateNets(exs []Example, activation la.OP, a, b Network) (aCost, bCost float64) {
	var aDesired, aActual, bDesired, bActual [][]float64

	for _, e := range exs {
		aDesired = append(aDesired, e.GetOutput())
		aActual = append(aActual, a.Prop(e.GetInput(), activation))
		bDesired = append(bDesired, e.GetOutput())
		bActual = append(bActual, b.Prop(e.GetInput(), activation))
	}

	return cost(aDesired, aActual), cost(bDesired, bActual)
}

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

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func sigmoidPrime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

func cost(desired, actual [][]float64) float64 {

	intermediate := make([]float64, len(desired))

	for i, _ := range desired {
		yMinA := la.VSUB(desired[i], actual[i])
		intermediate[i] = la.AddReduce(la.VMULT(yMinA, yMinA))
	}

	return la.AddReduce(intermediate) / float64(2*len(desired))
}

func costPrime(actual, desired float64) float64 {
	return actual - desired
}
