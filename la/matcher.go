package la

type Matcher interface {
	Match(a, b []float64) bool
}
