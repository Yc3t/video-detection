

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import PersonTrackingPipeline
from loguru import logger


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def add_analytics_overlay(frame: np.ndarray, result, pipeline: PersonTrackingPipeline, 
                         frame_stats: dict) -> np.ndarray:
    """Add real-time analytics information overlay to frame."""
    overlay_frame = frame.copy()
    
    try:
        # Create info panel
        panel_height = 180
        panel_width = 350
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (0, 0, 0)  # Black background
        
        # Add semi-transparent background
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay_frame, 0.7, overlay, 0.3, 0, overlay_frame)
        
        # Add border
        cv2.rectangle(overlay_frame, (10, 10), (10 + panel_width, 10 + panel_height), (255, 255, 255), 2)
        
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # Title
        cv2.putText(overlay_frame, "TRACKING ANALYTICS", (20, y_offset), font, 0.7, (0, 255, 255), thickness)
        y_offset += 25
        
        # Current frame statistics
        cv2.putText(overlay_frame, f"Detections: {len(result.detections)}", 
                   (20, y_offset), font, font_scale, color, 1)
        y_offset += 20
        
        cv2.putText(overlay_frame, f"Active Tracks: {len(result.tracks)}", 
                   (20, y_offset), font, font_scale, color, 1)
        y_offset += 20
        
        # Global statistics
        stats = pipeline.get_statistics()
        cv2.putText(overlay_frame, f"Total Created: {stats.get('total_tracks_created', 0)}", 
                   (20, y_offset), font, font_scale, color, 1)
        y_offset += 20
        
        # Track state breakdown
        confirmed_tracks = sum(1 for t in result.tracks if t.state == "confirmed")
        tentative_tracks = sum(1 for t in result.tracks if t.state == "tentative")
        
        cv2.putText(overlay_frame, f"Confirmed: {confirmed_tracks} | Tentative: {tentative_tracks}", 
                   (20, y_offset), font, 0.5, (255, 255, 0), 1)
        y_offset += 20
        
        # Processing performance
        processing_fps = 1.0 / result.processing_time if result.processing_time > 0 else 0
        cv2.putText(overlay_frame, f"Processing FPS: {processing_fps:.1f}", 
                   (20, y_offset), font, font_scale, (0, 255, 0), 1)
        y_offset += 20
        
        # Overall FPS
        if 'start_time' in frame_stats and frame_stats['frame_count'] > 0:
            elapsed = time.time() - frame_stats['start_time']
            overall_fps = frame_stats['frame_count'] / elapsed if elapsed > 0 else 0
            cv2.putText(overlay_frame, f"Overall FPS: {overall_fps:.1f}", 
                       (20, y_offset), font, font_scale, (0, 255, 0), 1)
        
        return overlay_frame
        
    except Exception as e:
        logger.warning(f"Error adding analytics overlay: {e}")
        return frame


def demo_webcam(pipeline: PersonTrackingPipeline, max_frames: int = 1000, show_analytics: bool = False):
    """Demo with webcam input."""
    logger.info("Starting webcam demo. Press 'q' to quit.")
    
    # Initialize frame statistics
    frame_stats = {
        'start_time': time.time(),
        'frame_count': 0
    }
    
    try:
        frame_count = 0
        for result in pipeline.process_stream(source=0, max_frames=max_frames):
            frame_count += 1
            frame_stats['frame_count'] = frame_count
            
            # Log progress every 30 frames
            if frame_count % 30 == 0:
                logger.info(
                    f"Frame {result.frame_id}: "
                    f"{len(result.detections)} detections, "
                    f"{len(result.tracks)} active tracks, "
                    f"{result.processing_time:.3f}s"
                )
            
            # Show analytics overlay if enabled
            if show_analytics:
                # so this is mainly for video processing
                pass
            
            # Break if max frames reached
            if frame_count >= max_frames:
                logger.info(f"Reached maximum frames ({max_frames})")
                break
        
        # Print final statistics
        stats = pipeline.get_statistics()
        logger.info("=== Final Statistics ===")
        logger.info(f"Total frames processed: {stats['total_frames_processed']}")
        logger.info(f"Total detections: {stats['total_detections']}")
        logger.info(f"Total tracks created: {stats['total_tracks_created']}")
        logger.info(f"Average processing time: {stats['average_processing_time']:.3f}s")
        logger.info(f"Estimated FPS: {1.0 / stats['average_processing_time']:.1f}")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error during webcam demo: {e}")


def demo_video(pipeline: PersonTrackingPipeline, video_path: str, output_path: str = None, 
               visualize: bool = True, preserve_audio: bool = True, show_analytics: bool = False):
    """Demo with video file input."""
    logger.info(f"Processing video: {video_path}")
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    try:
        # Set output path if not provided
        if output_path is None and visualize:
            video_file = Path(video_path)
            output_path = str(video_file.parent / f"{video_file.stem}_tracked{video_file.suffix}")
        
        # Enhanced processing with analytics overlay
        if show_analytics:
            demo_video_with_analytics(pipeline, video_path, output_path, preserve_audio)
            return
        
        # Process video
        result = pipeline.process_video(
            video_path=video_path,
            output_path=output_path,
            visualize=visualize,
            save_results=True,
            preserve_audio=preserve_audio
        )
        
        # Print summary
        logger.info("=== Processing Complete ===")
        logger.info(f"Video: {result.video_path}")
        logger.info(f"Duration: {result.duration:.2f}s ({result.total_frames} frames at {result.fps:.1f} FPS)")
        logger.info(f"Total detections: {result.summary.get('total_detections', 0)}")
        logger.info(f"Total tracks: {result.summary.get('total_tracks', 0)}")
        logger.info(f"Average detections per frame: {result.summary.get('average_detections_per_frame', 0):.1f}")
        logger.info(f"Average processing time: {result.summary.get('average_processing_time', 0):.3f}s")
        logger.info(f"Processing FPS: {result.summary.get('estimated_fps', 0):.1f}")
        
        if visualize and output_path:
            logger.info(f"Visualization saved to: {output_path}")
        
        # Track duration statistics
        track_durations = result.summary.get('track_durations', {})
        if track_durations:
            logger.info(f"Longest track duration: {result.summary.get('longest_track_duration', 0):.2f}s")
            logger.info(f"Average track duration: {result.summary.get('average_track_duration', 0):.2f}s")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")


def demo_video_with_analytics(pipeline: PersonTrackingPipeline, video_path: str, 
                              output_path: str, preserve_audio: bool = True):
    """Demo video processing with enhanced analytics overlay."""
    logger.info("Processing video with analytics overlay...")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video
    temp_video_path = None
    if preserve_audio:
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        video_output_path = temp_video_path
    else:
        video_output_path = output_path
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize frame statistics
    frame_stats = {
        'start_time': time.time(),
        'frame_count': 0
    }
    
    frame_id = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_id / fps
            frame_stats['frame_count'] = frame_id + 1
            
            # Process frame
            result = pipeline.process_frame(frame, frame_id, timestamp)
            
            # Create visualization
            vis_frame = pipeline.visualize_frame(frame, result)
            
            # Add analytics overlay
            vis_frame = add_analytics_overlay(vis_frame, result, pipeline, frame_stats)
            
            # Write frame
            out.write(vis_frame)
            
            # Show progress
            if frame_id % 100 == 0:
                progress = (frame_id / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% - Frame {frame_id}/{total_frames}")
            
            frame_id += 1
        
    finally:
        cap.release()
        out.release()
    
    # Combine with audio if needed
    if preserve_audio and temp_video_path:
        try:
            pipeline._combine_video_with_audio(video_path, temp_video_path, output_path)
            import os
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            logger.info("Audio preserved in output video")
        except Exception as e:
            logger.warning(f"Failed to preserve audio: {e}")
            # Remove existing output file if it exists and rename temp file
            if os.path.exists(temp_video_path):
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(temp_video_path, output_path)
                    logger.info("Video saved without audio (ffmpeg failed)")
                except Exception as rename_error:
                    logger.error(f"Failed to save video file: {rename_error}")
                    logger.info(f"Temporary video file available at: {temp_video_path}")
    
    logger.info(f"Video with analytics saved to: {output_path}")


def demo_image(pipeline: PersonTrackingPipeline, image_path: str, output_path: str = None):
    """Demo with single image input."""
    logger.info(f"Processing image: {image_path}")
    
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        return
    
    try:
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        logger.info(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Process frame
        result = pipeline.process_frame(frame, frame_id=0, timestamp=0.0)
        
        # Print results
        logger.info("=== Detection Results ===")
        logger.info(f"Detections: {len(result.detections)}")
        logger.info(f"Processing time: {result.processing_time:.3f}s")
        
        for i, detection in enumerate(result.detections):
            x1, y1, x2, y2 = detection.bbox
            logger.info(f"  Detection {i+1}: bbox=({x1}, {y1}, {x2}, {y2}), confidence={detection.confidence:.3f}")
        
        # Create visualization
        vis_frame = pipeline.visualize_frame(frame, result)
        
        # Save result
        if output_path is None:
            output_path = str(Path(image_path).with_suffix('_detected.jpg'))
        
        cv2.imwrite(output_path, vis_frame)
        logger.info(f"Result saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Person Detection and Tracking Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam demo
  python run_demo.py webcam
  
  # Webcam demo with analytics overlay
  python run_demo.py webcam --show-analytics
  
  # Process video file
  python run_demo.py video path/to/video.mp4
  
  # Process video with analytics overlay
  python run_demo.py video path/to/video.mp4 --show-analytics
  
  # Process single image
  python run_demo.py image path/to/image.jpg
  
  # Process video without visualization (faster)
  python run_demo.py video path/to/video.mp4 --no-visualize
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["webcam", "video", "image"],
        help="Input mode: webcam, video file, or single image"
    )
    
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to input file (required for video/image modes)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for results"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Maximum frames to process (webcam mode)"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization (video mode only)"
    )
    
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Don't preserve original audio in output video"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Compute device to use: auto, cpu, or gpu"
    )
    
    parser.add_argument(
        "--show-analytics",
        action="store_true",
        help="Show real-time analytics overlay on video output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if args.mode in ["video", "image"] and not args.input_path:
        logger.error(f"{args.mode} mode requires an input path")
        sys.exit(1)
    
    # Initialize pipeline
    logger.info("Initializing person tracking pipeline...")
    try:
        pipeline = PersonTrackingPipeline(config_path=args.config, device=args.device)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Run demo based on mode
    try:
        if args.mode == "webcam":
            demo_webcam(pipeline, args.max_frames, args.show_analytics)
        elif args.mode == "video":
            demo_video(
                pipeline, 
                args.input_path, 
                args.output, 
                not args.no_visualize,
                not args.no_audio,
                args.show_analytics
            )
        elif args.mode == "image":
            demo_image(pipeline, args.input_path, args.output)
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main() 