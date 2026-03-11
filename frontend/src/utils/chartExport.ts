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

function getHtmlElementSize(element: HTMLElement): { width: number; height: number } {
  const rect = element.getBoundingClientRect();
  const width = Math.max(1, Math.ceil(Math.max(rect.width, element.scrollWidth)));
  const height = Math.max(1, Math.ceil(Math.max(rect.height, element.scrollHeight)));
  return { width, height };
}

function collectDocumentCss(): string {
  let css = '';
  for (const styleSheet of Array.from(document.styleSheets)) {
    try {
      const rules = (styleSheet as CSSStyleSheet).cssRules;
      if (!rules) continue;
      for (const rule of Array.from(rules)) {
        css += `${rule.cssText}\n`;
      }
    } catch {
      // Ignore cross-origin stylesheets that cannot be read.
    }
  }
  return css;
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
  const { width, height } = getHtmlElementSize(element);
  const scale = options.scale ?? 3;
  const backgroundColor = options.backgroundColor ?? '#ffffff';
  const cssText = collectDocumentCss();

  const clone = element.cloneNode(true) as HTMLElement;
  clone.style.margin = '0';
  clone.style.width = `${width}px`;
  clone.style.height = `${height}px`;

  const serializedNode = new XMLSerializer().serializeToString(clone);
  const xhtml = `
    <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
      <foreignObject width="100%" height="100%">
        <div xmlns="http://www.w3.org/1999/xhtml" style="width:${width}px;height:${height}px;background:${backgroundColor};">
          <style>${cssText}</style>
          ${serializedNode}
        </div>
      </foreignObject>
    </svg>
  `;

  const blob = new Blob([xhtml], { type: 'image/svg+xml;charset=utf-8' });
  const objectUrl = URL.createObjectURL(blob);

  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load HTML snapshot for export'));
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
