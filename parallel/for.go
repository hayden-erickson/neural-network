package parallel

type Empty struct{}

type Semaphore chan Empty

func (s Semaphore) Signal() {
	s <- Empty{}
}

func (s Semaphore) Wait(n int) {
	for i := 0; i < n; i++ {
		<-s
	}
}

func For(data []float64, branch int, op func(float64) float64) []float64 {
	out := make([]float64, len(data))
	parHelper(data, out, 0, len(data), branch, op)
	return out
}

func parHelper(data, out []float64, start, fin, branch int, op func(float64) float64) {

	s := make(Semaphore)

	if fin-start <= 0 {
		return
	}

	if branch <= 0 {
		for i := start; i < fin; i++ {
			out[i] = op(data[i])
		}

		return
	}

	go func() {
		parHelper(data, out, start, fin/2, branch-1, op)
		s.Signal()
	}()

	parHelper(data, out, fin/2, fin, branch-1, op)
	s.Wait(1)

	close(s)
}
