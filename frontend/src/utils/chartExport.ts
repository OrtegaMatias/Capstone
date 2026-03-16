import html2canvas from 'html2canvas';

export type ExportPngOptions = {
  scale?: number;
  backgroundColor?: string;
};

function normalizeFilename(base: string): string {
  return base
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function triggerDownload(dataUrl: string, filename: string): void {
  const anchor = document.createElement('a');
  anchor.href = dataUrl;
  anchor.download = filename.endsWith('.png') ? filename : `${filename}.png`;
  anchor.click();
}

function getSvgSize(svg: SVGSVGElement): { width: number; height: number } {
  const rect = svg.getBoundingClientRect();
  const width = Math.max(1, Math.ceil(rect.width));
  const height = Math.max(1, Math.ceil(rect.height));
  return { width, height };
}

export async function exportSvgInContainerAsPng(
  container: HTMLElement,
  fileBaseName: string,
  options: ExportPngOptions = {},
): Promise<void> {
  const svg = container.querySelector('svg.main-svg') ?? container.querySelector('svg');
  if (!svg) {
    throw new Error('No SVG chart found in container');
  }

  const svgElement = svg as SVGSVGElement;
  const { width, height } = getSvgSize(svgElement);
  const scale = options.scale ?? 3;
  const backgroundColor = options.backgroundColor ?? '#ffffff';

  const clone = svgElement.cloneNode(true) as SVGSVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
  clone.setAttribute('width', `${width}`);
  clone.setAttribute('height', `${height}`);
  if (!clone.getAttribute('viewBox')) {
    clone.setAttribute('viewBox', `0 0 ${width} ${height}`);
  }

  const serialized = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([serialized], { type: 'image/svg+xml;charset=utf-8' });
  const objectUrl = URL.createObjectURL(blob);

  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load serialized SVG'));
      img.src = objectUrl;
    });

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Canvas 2D context not available');
    }

    ctx.scale(scale, scale);
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);

    const dataUrl = canvas.toDataURL('image/png');
    triggerDownload(dataUrl, normalizeFilename(fileBaseName));
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export async function exportHtmlElementAsPng(
  element: HTMLElement,
  fileBaseName: string,
  options: ExportPngOptions = {},
): Promise<void> {
  const scale = options.scale ?? 3;
  const backgroundColor = options.backgroundColor ?? '#ffffff';
  const canvas = await html2canvas(element, {
    scale,
    useCORS: true,
    allowTaint: true,
    backgroundColor,
    logging: false,
    windowWidth: Math.max(element.scrollWidth, element.clientWidth),
    windowHeight: Math.max(element.scrollHeight, element.clientHeight),
  });

  const dataUrl = canvas.toDataURL('image/png');
  triggerDownload(dataUrl, normalizeFilename(fileBaseName));
}
