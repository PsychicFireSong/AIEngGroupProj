export interface SampleImage {
  label: string;
  filename: string;
  defectHint: string;
  type: "image" | "video";
}

export const SAMPLE_IMAGES: SampleImage[] = [
  { label: "Crack", filename: "crack_sample.jpg", defectHint: "crack", type: "image" },
  { label: "Crack (video)", filename: "crack_source_j2PlmMnwKk8.mp4", defectHint: "crack", type: "video" },
  { label: "Spalling", filename: "spalling_sample.jpg", defectHint: "spalling", type: "image" },
  { label: "Spalling (video)", filename: "spalling_source_SaZO4udegs4.mp4", defectHint: "spalling", type: "video" },
  { label: "Corrosion", filename: "corrosion_sample.jpg", defectHint: "corrosion", type: "image" },
  { label: "Corrosion (video)", filename: "corrosion_source_110L1IZ0woo.mp4", defectHint: "corrosion", type: "video" },
  { label: "Pothole", filename: "pothole_sample.jpg", defectHint: "pothole", type: "image" },
  { label: "Pothole (video)", filename: "pothole_source_xPx4KTJ9ZAY.mp4", defectHint: "pothole", type: "video" },
  { label: "Paint Degradation", filename: "paint_degradation_sample.jpg", defectHint: "paint degradation", type: "image" },
  { label: "Paint Degradation (video)", filename: "paint_degradation_source_4bERZ7lPNh4.mp4", defectHint: "paint degradation", type: "video" },
  { label: "All defects", filename: "multi_defect_sample.jpg", defectHint: "all 5 classes", type: "image" },
  { label: "All defects", filename: "demo_video.mp4", defectHint: "all defect types", type: "video" },
];
