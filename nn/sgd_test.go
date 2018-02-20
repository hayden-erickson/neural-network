package nn_test

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/hayden-erickson/neural-network/la"
	"github.com/hayden-erickson/neural-network/loaders"
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
	var original Network
	var examples []Example
	var sgd SGD

	BeforeEach(func() {
		original, _ = NewNetwork([]int{16, 4})

		sgd = SGD{
			Activation: Sigmoid,
			Cost:       Quadratic,
			Eta:        5,
			Net:        original,
		}

		rand.Seed(time.Now().UnixNano())

		examples = generateExamples((rand.Intn(50) * 10) + 1000)

	})

	Describe("#MRun", func() {
		It("lowers the cost of the network", func() {
			// origCost := evaluateNet(examples, ToOP(Sigmoid.Fn), sgd.Net)
			var newCorrect int
			var newCost float64
			origCorrect, origCost := Evaluate(sgd, examples, binaryMatcher{})

			for i := 0; i < 10; i++ {
				sgd.MRun(examples, 50, 100)
				// newCost := evaluateNet(examples, ToOP(Sigmoid.Fn), sgd.Net)
				newCorrect, newCost = Evaluate(sgd, examples, binaryMatcher{})

				Expect(newCost).To(BeNumerically(`<`, origCost))
				origCost = newCost
			}

			Expect(origCorrect).To(BeNumerically(`<`, newCorrect))
		})

		It("continually changes the network", func() {
			oW := make([]la.Matrix, len(sgd.Net.Weights))
			oB := make([][]float64, len(sgd.Net.Biases))

			for i := 0; i < 10; i++ {
				copy(oW, sgd.Net.Weights)
				copy(oB, sgd.Net.Biases)

				sgd.MRun(examples, 50, 100)

				Expect(sgd.Net.Weights).ToNot(Equal(oW))
				Expect(sgd.Net.Biases).ToNot(Equal(oB))
			}
		})
	})

	Describe("#Run", func() {
		It("lowers the cost of the network", func() {
			for i := 0; i < 5; i++ {
				origCost := evaluateNet(examples, ToOP(Sigmoid.Fn), sgd.Net)
				sgd.Run(examples, 50, 100)
				newCost := evaluateNet(examples, ToOP(Sigmoid.Fn), sgd.Net)

				Expect(newCost).To(BeNumerically(`<`, origCost))
			}
		})

	})

})

func getData() []Example {
	l := loaders.NewMNISTLoader(
		`../data/t10k-images-idx3-ubyte`,
		`../data/t10k-labels-idx1-ubyte`,
		2051,
		2049)

	data, _ := l.Load()
	return data
}

func getSGD() SGD {
	net, _ := NewNetwork([]int{784, 30, 10})

	return SGD{
		Activation: Sigmoid,
		Cost:       Quadratic,
		Eta:        0.5,
		Net:        net,
	}
}

func BenchmarkRun(b *testing.B) {
	// sgd := nn.SGD{}
	sgd := getSGD()
	data := getData()

	for i := 0; i < b.N; i++ {
		sgd.Run(data[0:100], 1, 10)
	}
}

func BenchmarkMRun(b *testing.B) {
	sgd := getSGD()
	data := getData()

	for i := 0; i < b.N; i++ {
		sgd.MRun(data[0:100], 1, 10)
	}
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

func evaluateNet(exs []Example, activation la.OP, n Network) float64 {
	var desired, actual [][]float64

	for _, e := range exs {
		desired = append(desired, e.GetOutput())
		actual = append(actual, n.Prop(e.GetInput(), Sigmoid))
	}

	return cost(desired, actual)
}

func cost(desired, actual [][]float64) float64 {

	intermediate := make([]float64, len(desired))

	for i, _ := range desired {
		yMinA := la.VSUB(desired[i], actual[i])
		intermediate[i] = la.AddReduce(la.VMULT(yMinA, yMinA))
	}

	return la.AddReduce(intermediate) / float64(2*len(desired))
}

type binaryMatcher struct{}

func (bm binaryMatcher) Match(a, b []float64) bool {
	for i := 0; i < len(a); i++ {
		if math.Abs(a[i]-b[i]) > .025 {
			return false
		}
	}

	return true
}
