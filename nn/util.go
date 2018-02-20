package nn

import (
	"math"

	"github.com/hayden-erickson/neural-network/la"
)

type Differentiable interface {
	Fn(...float64) float64
	Prime(...float64) float64
}

type sigmoid struct{}

func (s sigmoid) Fn(zs ...float64) float64 {
	return 1 / (1 + math.Exp(-zs[0]))
}

func (s sigmoid) Prime(zs ...float64) float64 {
	return s.Fn(zs[0]) * (1 - s.Fn(zs[0]))
}

type quadratic struct{}

func (q quadratic) Fn(as ...float64) float64 {
	actual := as[0]
	desired := as[1]
	return math.Pow(desired-actual, 2)

	// for i, _ := range desired {
	// 	yMinA := la.VSUB(desired[i], actual[i])
	// 	intermediate[i] = la.AddReduce(la.VMULT(yMinA, yMinA))
	// }

	// return la.AddReduce(intermediate) / float64(2*len(desired))
}

func (q quadratic) Prime(as ...float64) float64 {
	return (as[0] - as[1])
}

type crossEntropy struct{}

func (ce crossEntropy) Fn(zs ...float64) float64 {
	actual := zs[0]
	desired := zs[1]
	// return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
	return -desired*math.Log(actual) - (1-desired)*math.Log(1-actual)
}

func (ce crossEntropy) Prime(zs ...float64) float64 {
	return zs[0] - zs[1]
}

func ToOP(f func(...float64) float64) la.OP {
	return func(z float64) float64 {
		return f(z)
	}
}

func ToBOP(f func(...float64) float64) la.BOP {
	return func(a, b float64) float64 {
		return f(a, b)
	}
}

var Sigmoid = sigmoid{}
var CrossEntropy = crossEntropy{}
var Quadratic = quadratic{}
