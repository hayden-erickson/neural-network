package la

type OP func(a float64) float64
type BOP func(a, b float64) float64

type Mapper func([]float64) []float64
type Aggregator func(a, b []float64) []float64
type MatrixAggregator func(a, b Matrix) Matrix
type MatrixMapper func(Matrix) Matrix

func Map(a []float64, op OP) []float64 {
	out := make([]float64, len(a))

	for i, _ := range a {
		out[i] = op(a[i])
	}

	return out
}

func MMap(a Matrix, op OP) Matrix {
	a.Data = Map(a.Data, op)
	return a
}

func MultBy(b float64) OP {
	return func(a float64) float64 {
		return a * b
	}
}

func Agg(a, b []float64, op BOP) []float64 {
	out := make([]float64, len(a))

	for i, _ := range a {
		out[i] = op(a[i], b[i])
	}

	return out
}

func Reduce(a []float64, op BOP) float64 {
	out := a[0]

	for i := 1; i < len(a); i++ {
		out = op(out, a[i])
	}

	return out
}

func createAgg(op BOP) Aggregator {
	return func(a, b []float64) []float64 {
		out := make([]float64, len(a))

		for i, _ := range a {
			out[i] = op(a[i], b[i])
		}

		return out
	}
}

func CreateMatrixAgg(op BOP) MatrixAggregator {
	return func(a, b Matrix) Matrix {
		a.Data = Agg(a.Data, b.Data, op)
		return a
	}
}

func createReduce(op BOP) func([]float64) float64 {
	return func(a []float64) float64 {
		return Reduce(a, op)
	}
}

func Add(x float64) OP {
	return func(a float64) float64 {
		return a + x
	}
}

func sum(a, b float64) float64 {
	return a + b
}

func sub(a, b float64) float64 {
	return a - b
}

func mult(a, b float64) float64 {
	return a * b
}

// vector operations
var SUM = createAgg(sum)
var SUB = createAgg(sub)
var MULT = createAgg(mult)
var AddReduce = createReduce(sum)

// matrix operations
var MSUM = CreateMatrixAgg(sum)
var MMULT = CreateMatrixAgg(mult)
