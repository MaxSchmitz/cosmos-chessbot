#!/bin/bash
# Test LiveChess2FEN with our chess board images

cd /Users/max/Code/cosmos-chessbot/external/LiveChess2FEN

# Test with one of our images
echo "Testing LiveChess2FEN with our chess board images..."
echo ""

# Test image 1: Simple e4 position
IMAGE1="/Users/max/Code/cosmos-chessbot/data/raw/rnbqkbnr:pppppppp:8:8:4P3:8:PPPP1PPP:RNBQKBNR b KQkq e3 0 1.png"
EXPECTED1="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

if [ -f "$IMAGE1" ]; then
    echo "Test 1: $(basename "$IMAGE1")"
    echo "Expected: $EXPECTED1"
    echo "Detected:"
    python lc2fen.py "$IMAGE1" BL --onnx
    echo ""
else
    echo "Image not found: $IMAGE1"
fi

echo "Note: LiveChess2FEN requires:"
echo "- Board detection to find chessboard in image"
echo "- Correct a1_pos parameter (BL=bottom-left, BR=bottom-right, TL=top-left, TR=top-right)"
echo "- May need adjustment for egocentric camera angles"
