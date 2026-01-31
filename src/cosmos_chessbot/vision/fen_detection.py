"""FEN detection using Ultimate V2 board segmentation + YOLO piece detection.

This module combines two models for complete chess FEN detection:
1. Ultimate V2: Fast board segmentation (15ms)
2. YOLO: Accurate piece detection

Pipeline:
Image → [Ultimate V2] → Cropped Board → [YOLO] → Piece Positions → FEN
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .board_segmentation import BoardSegmentation


@dataclass
class ChessPiece:
    """Detected chess piece."""
    piece_type: str  # 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king'
    color: str  # 'white' or 'black'
    x: float  # Center x coordinate
    y: float  # Center y coordinate
    confidence: float  # Detection confidence


class FENDetector:
    """Complete FEN detection using board segmentation + piece detection.

    Uses:
    - Ultimate V2 ONNX model for board segmentation
    - YOLO model for piece detection
    - Geometric mapping to convert piece positions to FEN
    """

    # FEN piece notation
    PIECE_TO_FEN = {
        ('pawn', 'white'): 'P',
        ('pawn', 'black'): 'p',
        ('rook', 'white'): 'R',
        ('rook', 'black'): 'r',
        ('knight', 'white'): 'N',
        ('knight', 'black'): 'n',
        ('bishop', 'white'): 'B',
        ('bishop', 'black'): 'b',
        ('queen', 'white'): 'Q',
        ('queen', 'black'): 'q',
        ('king', 'white'): 'K',
        ('king', 'black'): 'k',
    }

    def __init__(
        self,
        board_seg_model_path: Optional[Path] = None,
        piece_det_model_path: Optional[Path] = None,
    ):
        """Initialize FEN detector.

        Args:
            board_seg_model_path: Path to Ultimate V2 ONNX model
            piece_det_model_path: Path to YOLO piece detection model
        """
        # Initialize board segmentation
        self.board_seg = BoardSegmentation(model_path=board_seg_model_path)

        # Initialize piece detection
        self.piece_detector = self._load_piece_detector(piece_det_model_path)

    def _load_piece_detector(self, model_path: Optional[Path]):
        """Load YOLO piece detection model.

        Args:
            model_path: Path to YOLO model

        Returns:
            Loaded YOLO model
        """
        try:
            from ultralytics import YOLO

            # Default model path
            if model_path is None:
                model_path = Path(__file__).parent.parent.parent.parent / "models" / "chess_piece_yolo.pt"

            model = YOLO(str(model_path))
            print(f"Loaded YOLO piece detection model from {model_path}")
            return model

        except ImportError:
            raise ImportError(
                "Ultralytics YOLO not found. Install with: pip install ultralytics"
            )

    def detect_fen(
        self,
        image: Image.Image,
        active_color: str = 'w',
        castling: str = 'KQkq',
        en_passant: str = '-',
    ) -> str:
        """Detect FEN from chess board image.

        Args:
            image: PIL Image of chess board
            active_color: Active color ('w' or 'b')
            castling: Castling availability (e.g., 'KQkq' or '-')
            en_passant: En passant target square (e.g., 'e3' or '-')

        Returns:
            Complete FEN string
        """
        # Step 1: Segment board
        mask, cropped_board = self.board_seg.segment_board(image)

        # Step 2: Detect pieces on cropped board
        pieces = self._detect_pieces(cropped_board)

        # Step 3: Map pieces to squares
        board_array = self._pieces_to_board_array(pieces, cropped_board.shape)

        # Step 4: Convert to FEN
        position_fen = self._board_array_to_fen(board_array)

        # Step 5: Assemble complete FEN
        complete_fen = f"{position_fen} {active_color} {castling} {en_passant} 0 1"

        return complete_fen

    def _detect_pieces(self, board_image: np.ndarray) -> List[ChessPiece]:
        """Detect chess pieces using YOLO.

        Args:
            board_image: Cropped board image (H, W, C)

        Returns:
            List of detected ChessPiece objects
        """
        # Run YOLO inference
        results = self.piece_detector(board_image, verbose=False)

        pieces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Parse class name (format: "white_pawn", "black_rook", etc.)
                class_name = result.names[cls]
                parts = class_name.split('_')
                if len(parts) == 2:
                    color, piece_type = parts
                else:
                    # Fallback parsing
                    color = 'white' if 'white' in class_name.lower() else 'black'
                    piece_type = class_name.lower().replace('white_', '').replace('black_', '')

                # Calculate center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                pieces.append(ChessPiece(
                    piece_type=piece_type,
                    color=color,
                    x=center_x,
                    y=center_y,
                    confidence=conf,
                ))

        return pieces

    def _pieces_to_board_array(
        self,
        pieces: List[ChessPiece],
        image_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Map detected pieces to 8x8 board array.

        Args:
            pieces: List of detected pieces
            image_shape: Shape of board image (H, W, C)

        Returns:
            8x8 array with FEN piece notation (or None for empty squares)
        """
        board = np.full((8, 8), None, dtype=object)

        height, width = image_shape[:2]
        square_width = width / 8
        square_height = height / 8

        for piece in pieces:
            # Convert pixel coordinates to board coordinates
            file = int(piece.x / square_width)  # Column (a-h)
            rank = int(piece.y / square_height)  # Row (8-1 from top)

            # Clamp to valid range
            file = max(0, min(7, file))
            rank = max(0, min(7, rank))

            # Get FEN notation
            fen_char = self.PIECE_TO_FEN.get((piece.piece_type, piece.color))
            if fen_char:
                board[rank, file] = fen_char

        return board

    def _board_array_to_fen(self, board: np.ndarray) -> str:
        """Convert 8x8 board array to FEN position string.

        Args:
            board: 8x8 array with piece notation

        Returns:
            FEN position string (first part of FEN)
        """
        fen_ranks = []

        for rank in board:
            fen_rank = ""
            empty_count = 0

            for square in rank:
                if square is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_rank += str(empty_count)
                        empty_count = 0
                    fen_rank += square

            # Add remaining empty squares
            if empty_count > 0:
                fen_rank += str(empty_count)

            fen_ranks.append(fen_rank)

        return '/'.join(fen_ranks)

    def visualize_detection(
        self,
        image: Image.Image,
        save_path: Optional[Path] = None,
    ) -> Tuple[Image.Image, str]:
        """Visualize FEN detection results.

        Args:
            image: Input PIL Image
            save_path: Optional path to save visualization

        Returns:
            Tuple of (visualization image, detected FEN)
        """
        import matplotlib.pyplot as plt

        # Detect FEN
        fen = self.detect_fen(image)

        # Get segmentation and detection
        mask, cropped_board = self.board_seg.segment_board(image)
        pieces = self._detect_pieces(cropped_board)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Segmentation
        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title('Board Segmentation')
        axes[1].axis('off')

        # Piece detection
        axes[2].imshow(cropped_board)
        for piece in pieces:
            color = 'white' if piece.color == 'white' else 'yellow'
            axes[2].plot(piece.x, piece.y, 'o', color=color, markersize=8)
            axes[2].text(
                piece.x, piece.y - 10,
                f"{piece.piece_type[0].upper()}",
                color=color,
                fontsize=10,
                weight='bold'
            )
        axes[2].set_title(f'Piece Detection\nFEN: {fen}')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")

        # Convert to PIL Image
        fig.canvas.draw()
        vis_image = Image.frombytes(
            'RGB',
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb()
        )
        plt.close(fig)

        return vis_image, fen
