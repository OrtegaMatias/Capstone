import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

const MM_PER_INCH = 25.4;
const A4_WIDTH_MM = 210;
const A4_HEIGHT_MM = 297;
const PAGE_MARGIN_MM = 10;

function mmToPx(mm: number, dpi: number): number {
  return Math.max(1, Math.round((mm / MM_PER_INCH) * dpi));
}

async function waitForFonts(): Promise<void> {
  const fontSet = (document as Document & { fonts?: { ready?: Promise<unknown> } }).fonts;
  if (fontSet?.ready) {
    await fontSet.ready;
  }
}

function buildExportClone(element: HTMLElement): HTMLElement {
  const clone = element.cloneNode(true) as HTMLElement;
  clone.style.width = `${Math.max(element.scrollWidth, element.clientWidth)}px`;
  clone.style.maxWidth = 'none';
  clone.style.boxSizing = 'border-box';
  clone.style.background = '#ffffff';
  clone.style.paddingBottom = '24px';

  clone.querySelectorAll('details').forEach((detail) => {
    detail.setAttribute('open', 'true');
  });

  clone.querySelectorAll('button, .hero-actions, .outline-btn, [data-pdf-ignore="true"]').forEach((node) => {
    (node as HTMLElement).style.display = 'none';
  });

  clone.querySelectorAll('.report-preview').forEach((node) => {
    const pre = node as HTMLElement;
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.wordBreak = 'break-word';
    pre.style.overflow = 'visible';
    pre.style.maxHeight = 'none';
  });

  clone.querySelectorAll('.table-wrap').forEach((node) => {
    const tableWrap = node as HTMLElement;
    tableWrap.style.overflow = 'visible';
  });

  return clone;
}

function mountCloneForCapture(element: HTMLElement): { sandbox: HTMLDivElement; clone: HTMLElement } {
  const sandbox = document.createElement('div');
  sandbox.style.position = 'fixed';
  sandbox.style.left = '-100000px';
  sandbox.style.top = '0';
  sandbox.style.width = `${Math.max(element.scrollWidth, element.clientWidth)}px`;
  sandbox.style.pointerEvents = 'none';
  sandbox.style.opacity = '1';
  sandbox.style.zIndex = '-1';
  sandbox.style.background = '#ffffff';

  const clone = buildExportClone(element);
  sandbox.appendChild(clone);
  document.body.appendChild(sandbox);
  return { sandbox, clone };
}

function canvasSliceToDataUrl(source: HTMLCanvasElement, startY: number, height: number): string {
  const slice = document.createElement('canvas');
  slice.width = source.width;
  slice.height = height;
  const ctx = slice.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas 2D context not available for PDF slice');
  }
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, slice.width, slice.height);
  ctx.drawImage(source, 0, startY, source.width, height, 0, 0, source.width, height);
  return slice.toDataURL('image/png');
}

/**
 * Captures the full content of a DOM element and generates a paginated A4 PDF.
 * The export waits for fonts, clones the DOM offscreen, expands <details>,
 * hides controls, and slices the rendered canvas page by page to avoid blur and bad offsets.
 */
export async function downloadPageAsPdf(
  element: HTMLElement,
  filename: string,
  statusCallback?: (msg: string) => void,
): Promise<void> {
  statusCallback?.('Preparando contenido...');
  await waitForFonts();

  const { sandbox, clone } = mountCloneForCapture(element);

  try {
    statusCallback?.('Capturando páginas...');

    const captureScale = 2;
    const targetDpi = 96 * captureScale;
    const printableWidthMm = A4_WIDTH_MM - PAGE_MARGIN_MM * 2;
    const printableHeightMm = A4_HEIGHT_MM - PAGE_MARGIN_MM * 2;

    const canvas = await html2canvas(clone, {
      scale: captureScale,
      useCORS: true,
      allowTaint: true,
      backgroundColor: '#ffffff',
      logging: false,
      foreignObjectRendering: true,
      imageTimeout: 0,
      windowWidth: Math.max(clone.scrollWidth, clone.clientWidth),
      windowHeight: Math.max(clone.scrollHeight, clone.clientHeight),
      scrollX: 0,
      scrollY: 0,
    });

    statusCallback?.('Generando PDF...');

    const pdf = new jsPDF('p', 'mm', 'a4');
    const pageWidthPx = mmToPx(printableWidthMm, targetDpi);
    const pageHeightPx = mmToPx(printableHeightMm, targetDpi);

    const scaledPageHeightPx = Math.max(1, Math.floor((canvas.width * pageHeightPx) / pageWidthPx));
    const totalPages = Math.max(1, Math.ceil(canvas.height / scaledPageHeightPx));

    for (let pageIndex = 0; pageIndex < totalPages; pageIndex += 1) {
      const startY = pageIndex * scaledPageHeightPx;
      const sliceHeight = Math.min(scaledPageHeightPx, canvas.height - startY);
      const sliceDataUrl = canvasSliceToDataUrl(canvas, startY, sliceHeight);
      const renderedHeightMm = (sliceHeight * printableWidthMm) / canvas.width;

      if (pageIndex > 0) {
        pdf.addPage();
      }
      pdf.addImage(
        sliceDataUrl,
        'PNG',
        PAGE_MARGIN_MM,
        PAGE_MARGIN_MM,
        printableWidthMm,
        renderedHeightMm,
        undefined,
        'FAST',
      );
    }

    statusCallback?.('Descargando...');
    pdf.save(filename);
  } finally {
    sandbox.remove();
  }
}
