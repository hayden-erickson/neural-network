package loaders

type mnistMatcher struct{}

func (m mnistMatcher) Match(a, b []float64) bool {
	aMax := a[0]
	bMax := b[0]

	if len(a) != len(b) {
		panic(`proped input and output are mismatched`)
	}

	var aIdx, bIdx int

	for i, _ := range a {
		if a[i] > aMax {
			aMax = a[i]
			aIdx = i
		}

		if b[i] > bMax {
			bMax = b[i]
			bIdx = i
		}
	}

	return bIdx == aIdx
}

var MnistMatcher = mnistMatcher{}
