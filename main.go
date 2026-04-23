package main

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/notnil/chess"
)

// Piece values in 'centipawns' (100 = 1 pawn)
// To "train" your bot, you adjust these numbers based on performance.
var pieceValues = map[chess.PieceType]int{
	chess.Pawn:   100,
	chess.Knight: 320,
	chess.Bishop: 330,
	chess.Rook:   500,
	chess.Queen:  900,
	chess.King:   20000,
}

// Evaluation function: Simply sums up the material on the board.
func evaluate(board *chess.Board) int {
	score := 0
	for _, piece := range board.SquareMap() {
		val := pieceValues[piece.Type()]
		if piece.Color() == chess.White {
			score += val
		} else {
			score -= val
		}
	}
	return score
}

// Minimax with Alpha-Beta Pruning
func minimax(pos *chess.Position, depth int, alpha int, beta int, isMaximizing bool) int {
	if depth == 0 {
		return evaluate(pos.Board())
	}

	moves := pos.ValidMoves()
	if len(moves) == 0 {
		if pos.Status() == chess.Checkmate {
			if isMaximizing {
				return -30000 // Loss
			}
			return 30000 // Win
		}
		return 0 // Draw
	}

	if isMaximizing {
		maxEval := math.MinInt32
		for _, move := range moves {
			eval := minimax(pos.Update(move), depth-1, alpha, beta, false)
			if eval > maxEval {
				maxEval = eval
			}
			if eval > alpha {
				alpha = eval
			}
			if beta <= alpha {
				break
			}
		}
		return maxEval
	} else {
		minEval := math.MaxInt32
		for _, move := range moves {
			eval := minimax(pos.Update(move), depth-1, alpha, beta, true)
			if eval < minEval {
				minEval = eval
			}
			if eval < beta {
				beta = eval
			}
			if beta <= alpha {
				break
			}
		}
		return minEval
	}
}

type moveResult struct {
	move *chess.Move
	eval int
}

func main() {
	// Set Go to use all 3 cores (300% CPU)
	runtime.GOMAXPROCS(3)
	
	game := chess.NewGame()
	fmt.Println("--- Chess AI Started ---")
	fmt.Println("System: 3 Cores (300%) | 2GB RAM")

	for game.Outcome() == chess.NoOutcome {
		moves := game.ValidMoves()
		results := make(chan moveResult, len(moves))
		var wg sync.WaitGroup

		isWhite := game.Position().Turn() == chess.White

		// Distribute move calculations across all CPU cores
		for _, move := range moves {
			wg.Add(1)
			go func(m *chess.Move) {
				defer wg.Done()
				// Depth 4 is a great balance for 3 cores. 
				// Depth 5 might take a few seconds per move.
				eval := minimax(game.Position().Update(m), 4, math.MinInt32, math.MaxInt32, !isWhite)
				results <- moveResult{m, eval}
			}(move)
		}

		// Wait for all "workers" to finish
		wg.Wait()
		close(results)

		var bestMove *chess.Move
		if isWhite {
			bestEval := math.MinInt32
			for res := range results {
				if res.eval > bestEval || bestMove == nil {
					bestEval = res.eval
					bestMove = res.move
				}
			}
		} else {
			bestEval := math.MaxInt32
			for res := range results {
				if res.eval < bestEval || bestMove == nil {
					bestEval = res.eval
					bestMove = res.move
				}
			}
		}

		// Execute the move
		game.Move(bestMove)
		fmt.Println(game.Position().Board().Draw())
		fmt.Printf("Engine played: %s\n", bestMove)
		fmt.Println("---------------------------")
	}

	fmt.Printf("Game Over. Outcome: %s\n", game.Outcome())
}
