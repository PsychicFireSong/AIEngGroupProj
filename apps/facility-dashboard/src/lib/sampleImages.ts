// Drop your test images into public/samples/ and list them here.
// Each entry will appear as a clickable card in the monitoring page.
export type SampleImage = {
  label: string;       // shown on the card
  filename: string;    // file in public/samples/
  defectHint: string;  // small tag shown under label (e.g. "crack · corrosion")
};

export const SAMPLE_IMAGES: SampleImage[] = [
  { label: "Crack",             filename: "crack_sample.jpg",             defectHint: "crack" },
  { label: "Spalling",          filename: "spalling_sample.jpg",          defectHint: "spalling" },
  { label: "Corrosion",         filename: "corrosion_sample.jpg",         defectHint: "corrosion" },
  { label: "Pothole",           filename: "pothole_sample.jpg",           defectHint: "pothole" },
  { label: "Paint Degradation", filename: "paint_degradation_sample.jpg", defectHint: "paint degradation" },
  { label: "Multi-defect",      filename: "multi_defect_sample.jpg",      defectHint: "crack · spalling · corrosion" },
];
