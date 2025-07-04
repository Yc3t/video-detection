import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
import time

# añade src a path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import PersonTrackingPipeline
from loguru import logger


def setup_logging(verbose: bool = False):
    """configura logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def add_analytics_overlay(frame: np.ndarray, result, pipeline: PersonTrackingPipeline, 
                         frame_stats: dict) -> np.ndarray:
    """añade overlay información analytics tiempo real a frame."""
    overlay_frame = frame.copy()
    
    try:
        # crea panel info
        panel_height = 180
        panel_width = 350
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (0, 0, 0)  # fondo negro
        
        # añade fondo semi-transparente
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay_frame, 0.7, overlay, 0.3, 0, overlay_frame)
        
        # añade borde
        cv2.rectangle(overlay_frame, (10, 10), (10 + panel_width, 10 + panel_height), (255, 255, 255), 2)
        
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # título
        cv2.putText(overlay_frame, "analytics tracking", (20, y_offset), font, 0.7, (0, 255, 255), thickness)
        y_offset += 25
        
        # estadísticas frame actual
        cv2.putText(overlay_frame, f"detecciones: {len(result.detections)}", 
                   (20, y_offset), font, font_scale, color, 1)
        y_offset += 20
        
        cv2.putText(overlay_frame, f"tracks activos: {len(result.tracks)}", 
                   (20, y_offset), font, font_scale, color, 1)
        y_offset += 20
        
        # estadísticas globales
        stats = pipeline.get_statistics()
        cv2.putText(overlay_frame, f"total creados: {stats.get('total_tracks_created', 0)}", 
                   (20, y_offset), font, font_scale, color, 1)
        y_offset += 20
        
        confirmed_tracks = sum(2 for t in result.tracks if t.state == "confirmed")
        tentative_tracks = sum(1 for t in result.tracks if t.state == "tentative")
        
        cv2.putText(overlay_frame, f"confirmados: {confirmed_tracks} | tentativos: {tentative_tracks}", 
                   (20, y_offset), font, 0.5, (255, 255, 0), 1)
        y_offset += 20
        
        # rendimiento procesamiento
        processing_fps = 1.0 / result.processing_time if result.processing_time > 0 else 0
        cv2.putText(overlay_frame, f"fps procesamiento: {processing_fps:.1f}", 
                   (20, y_offset), font, font_scale, (0, 255, 0), 1)
        y_offset += 20
        
        # fps general
        if 'start_time' in frame_stats and frame_stats['frame_count'] > 0:
            elapsed = time.time() - frame_stats['start_time']
            overall_fps = frame_stats['frame_count'] / elapsed if elapsed > 0 else 0
            cv2.putText(overlay_frame, f"fps general: {overall_fps:.1f}", 
                       (20, y_offset), font, font_scale, (0, 255, 0), 1)
        
        return overlay_frame
        
    except Exception as e:
        logger.warning(f"error añadiendo overlay analytics: {e}")
        return frame


def demo_webcam(pipeline: PersonTrackingPipeline, max_frames: int = 1000, show_analytics: bool = False):
    """demo con entrada webcam."""
    logger.info("iniciando demo webcam. presiona 'q' para salir.")
    
    # inicializa estadísticas frame
    frame_stats = {
        'start_time': time.time(),
        'frame_count': 0
    }
    
    try:
        frame_count = 0
        for result in pipeline.process_stream(source=0, max_frames=max_frames):
            frame_count += 1
            frame_stats['frame_count'] = frame_count
            
            # log progreso cada 30 frames
            if frame_count % 30 == 0:
                logger.info(
                    f"frame {result.frame_id}: "
                    f"{len(result.detections)} detecciones, "
                    f"{len(result.tracks)} tracks activos, "
                    f"{result.processing_time:.3f}s"
                )
            
            # muestra overlay analytics si habilitado
            if show_analytics:
                # esto es principalmente para procesamiento video
                pass
            
            # termina si alcanza frames máximos
            if frame_count >= max_frames:
                logger.info(f"alcanzado frames máximos ({max_frames})")
                break
        
        # imprime estadísticas finales
        stats = pipeline.get_statistics()
        logger.info("=== estadísticas finales ===")
        logger.info(f"total frames procesados: {stats['total_frames_processed']}")
        logger.info(f"total detecciones: {stats['total_detections']}")
        logger.info(f"total tracks creados: {stats['total_tracks_created']}")
        logger.info(f"tiempo procesamiento promedio: {stats['average_processing_time']:.3f}s")
        logger.info(f"fps estimado: {1.0 / stats['average_processing_time']:.1f}")
        
    except KeyboardInterrupt:
        logger.info("demo interrumpido por usuario")
    except Exception as e:
        logger.error(f"error durante demo webcam: {e}")


def demo_video(pipeline: PersonTrackingPipeline, video_path: str, output_path: str = None, 
               visualize: bool = True, preserve_audio: bool = True, show_analytics: bool = False):
    """demo con entrada archivo video."""
    logger.info(f"procesando video: {video_path}")
    
    if not Path(video_path).exists():
        logger.error(f"archivo video no encontrado: {video_path}")
        return
    
    try:
        # establece path salida si no proporcionado
        if output_path is None and visualize:
            video_file = Path(video_path)
            output_path = str(video_file.parent / f"{video_file.stem}_tracked{video_file.suffix}")
        
        # procesamiento mejorado con overlay analytics
        if show_analytics:
            demo_video_with_analytics(pipeline, video_path, output_path, preserve_audio)
            return
        
        # procesa video
        result = pipeline.process_video(
            video_path=video_path,
            output_path=output_path,
            visualize=visualize,
            save_results=True,
            preserve_audio=preserve_audio
        )
        
        # imprime resumen
        logger.info("=== procesamiento completo ===")
        logger.info(f"video: {result.video_path}")
        logger.info(f"duración: {result.duration:.2f}s ({result.total_frames} frames a {result.fps:.1f} fps)")
        logger.info(f"total detecciones: {result.summary.get('total_detections', 0)}")
        logger.info(f"total tracks: {result.summary.get('total_tracks', 0)}")
        logger.info(f"detecciones promedio por frame: {result.summary.get('average_detections_per_frame', 0):.1f}")
        logger.info(f"tiempo procesamiento promedio: {result.summary.get('average_processing_time', 0):.3f}s")
        logger.info(f"fps procesamiento: {result.summary.get('estimated_fps', 0):.1f}")
        
        if visualize and output_path:
            logger.info(f"visualización guardada en: {output_path}")
        
        # estadísticas duración tracks
        track_durations = result.summary.get('track_durations', {})
        if track_durations:
            logger.info(f"duración track más largo: {result.summary.get('longest_track_duration', 0):.2f}s")
            logger.info(f"duración track promedio: {result.summary.get('average_track_duration', 0):.2f}s")
        
    except Exception as e:
        logger.error(f"error procesando video: {e}")


def demo_video_with_analytics(pipeline: PersonTrackingPipeline, video_path: str, 
                              output_path: str, preserve_audio: bool = True):
    """demo procesamiento video con overlay analytics mejorado."""
    logger.info("procesando video con overlay analytics...")
    
    # abre video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"no pudo abrir video: {video_path}")
    
    # obtiene propiedades video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # configura video salida
    temp_video_path = None
    if preserve_audio:
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        video_output_path = temp_video_path
    else:
        video_output_path = output_path
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))
    
    # inicializa estadísticas frame
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
            
            # procesa frame
            result = pipeline.process_frame(frame, frame_id, timestamp)
            
            # crea visualización
            vis_frame = pipeline.visualize_frame(frame, result)
            
            # añade overlay analytics
            vis_frame = add_analytics_overlay(vis_frame, result, pipeline, frame_stats)
            
            # escribe frame
            out.write(vis_frame)
            
            # muestra progreso
            if frame_id % 100 == 0:
                progress = (frame_id / total_frames) * 100
                logger.info(f"progreso: {progress:.1f}% - frame {frame_id}/{total_frames}")
            
            frame_id += 1
        
    finally:
        cap.release()
        out.release()
    
    # combina con audio si necesario
    if preserve_audio and temp_video_path:
        try:
            pipeline._combine_video_with_audio(video_path, temp_video_path, output_path)
            import os
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            logger.info("audio preservado en video salida")
        except Exception as e:
            logger.warning(f"fallo preservar audio: {e}")
            # remueve archivo salida existente si existe y renombra archivo temp
            if os.path.exists(temp_video_path):
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(temp_video_path, output_path)
                    logger.info("video guardado sin audio (ffmpeg falló)")
                except Exception as rename_error:
                    logger.error(f"fallo guardar archivo video: {rename_error}")
                    logger.info(f"archivo video temporal disponible en: {temp_video_path}")
    
    logger.info(f"video con analytics guardado en: {output_path}")


def demo_image(pipeline: PersonTrackingPipeline, image_path: str, output_path: str = None):
    """demo con entrada imagen single."""
    logger.info(f"procesando imagen: {image_path}")
    
    if not Path(image_path).exists():
        logger.error(f"archivo imagen no encontrado: {image_path}")
        return
    
    try:
        # carga imagen
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"no pudo cargar imagen: {image_path}")
            return
        
        logger.info(f"tamaño imagen: {frame.shape[1]}x{frame.shape[0]}")
        
        # procesa frame
        result = pipeline.process_frame(frame, frame_id=0, timestamp=0.0)
        
        # imprime resultados
        logger.info("=== resultados detección ===")
        logger.info(f"detecciones: {len(result.detections)}")
        logger.info(f"tiempo procesamiento: {result.processing_time:.3f}s")
        
        for i, detection in enumerate(result.detections):
            x1, y1, x2, y2 = detection.bbox
            logger.info(f"  detección {i+1}: bbox=({x1}, {y1}, {x2}, {y2}), confianza={detection.confidence:.3f}")
        
        # crea visualización
        vis_frame = pipeline.visualize_frame(frame, result)
        
        # guarda resultado
        if output_path is None:
            output_path = str(Path(image_path).with_suffix('_detectado.jpg'))
        
        cv2.imwrite(output_path, vis_frame)
        logger.info(f"resultado guardado en: {output_path}")
        
    except Exception as e:
        logger.error(f"error procesando imagen: {e}")


def main():
    """función principal."""
    parser = argparse.ArgumentParser(
        description="demo detección y tracking personas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ejemplos:
  # demo webcam
  python run_demo.py webcam
  
  # demo webcam con overlay analytics
  python run_demo.py webcam --show-analytics
  
  # procesa archivo video
  python run_demo.py video path/to/video.mp4
  
  # procesa video con overlay analytics
  python run_demo.py video path/to/video.mp4 --show-analytics
  
  # procesa imagen single
  python run_demo.py image path/to/image.jpg
  
  # procesa video sin visualización (más rápido)
  python run_demo.py video path/to/video.mp4 --no-visualize
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["webcam", "video", "image"],
        help="modo entrada: webcam, archivo video, o imagen single"
    )
    
    parser.add_argument(
        "input_path",
        nargs="?",
        help="path archivo entrada (requerido para modos video/imagen)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="path salida para resultados"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        help="path archivo configuración"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="frames máximos procesar (modo webcam)"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="deshabilita visualización (solo modo video)"
    )
    
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="no preserva audio original en video salida"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="habilita logging verbose"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="dispositivo computación usar: auto, cpu, o gpu"
    )
    
    parser.add_argument(
        "--show-analytics",
        action="store_true",
        help="muestra overlay analytics tiempo real en salida video"
    )
    
    args = parser.parse_args()
    
    # configura logging
    setup_logging(args.verbose)
    
    # valida argumentos
    if args.mode in ["video", "image"] and not args.input_path:
        logger.error(f"modo {args.mode} requiere path entrada")
        sys.exit(1)
    
    # inicializa pipeline
    logger.info("inicializando pipeline tracking personas...")
    try:
        pipeline = PersonTrackingPipeline(config_path=args.config, device=args.device)
        logger.info("pipeline inicializado exitosamente")
    except Exception as e:
        logger.error(f"fallo inicializar pipeline: {e}")
        sys.exit(1)
    
    # ejecuta demo basado en modo
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
        logger.info("demo interrumpido por usuario")
    except Exception as e:
        logger.error(f"demo falló: {e}")
        sys.exit(1)
    
    logger.info("demo completado exitosamente!")


if __name__ == "__main__":
    main() 