package falkenvector

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"
)

type eventEmitter struct {
	runID string
	sink  EventSink
	mu    sync.Mutex
	seq   uint64
	dead  bool
}

func newEventEmitter(sink EventSink) *eventEmitter {
	return &eventEmitter{
		runID: newRunID(),
		sink:  sink,
	}
}

func (e *eventEmitter) emit(event Event) {
	if e == nil {
		return
	}
	e.mu.Lock()
	if e.sink == nil || e.dead {
		e.mu.Unlock()
		return
	}
	e.seq++
	if event.At.IsZero() {
		event.At = time.Now().UTC()
	}
	if event.RunID == "" {
		event.RunID = e.runID
	}
	if event.Seq == 0 {
		event.Seq = e.seq
	}
	sink := e.sink
	e.mu.Unlock()

	defer func() {
		if recover() != nil {
			e.mu.Lock()
			e.dead = true
			e.sink = nil
			e.mu.Unlock()
		}
	}()
	sink(event)
}

func (e *eventEmitter) emitError(kind EventType, err error) {
	if err == nil {
		e.emit(Event{Type: kind})
		return
	}
	e.emit(Event{Type: kind, Error: err.Error()})
}

func newRunID() string {
	var raw [16]byte
	if _, err := rand.Read(raw[:]); err == nil {
		return "run_" + hex.EncodeToString(raw[:])
	}
	return "run_" + hex.EncodeToString([]byte(time.Now().UTC().Format(time.RFC3339Nano)))
}
