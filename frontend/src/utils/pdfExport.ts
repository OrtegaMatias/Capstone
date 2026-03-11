import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

/**
 * Captures the full visible content of a DOM element and generates a multi-page PDF.
 * Uses html2canvas to render the element as a high-resolution image,
 * then slices it into A4-sized pages using jsPDF.
 */
export async function downloadPageAsPdf(
  element: HTMLElement,
  filename: string,
  statusCallback?: (msg: string) => void,
): Promise<void> {
  statusCallback?.('Capturando pantalla...');

  const canvas = await html2canvas(element, {
    scale: 2,
    useCORS: true,
    allowTaint: true,
    backgroundColor: '#ffffff',
    logging: false,
    windowWidth: element.scrollWidth,
    windowHeight: element.scrollHeight,
  });

  statusCallback?.('Generando PDF...');

  const imgData = canvas.toDataURL('image/jpeg', 0.92);

  const pdfWidth = 210; // A4 width in mm
  const pdfHeight = 297; // A4 height in mm
  const imgWidth = pdfWidth;
  const imgHeight = (canvas.height * pdfWidth) / canvas.width;

  const pdf = new jsPDF('p', 'mm', 'a4');
  let heightLeft = imgHeight;
  let position = 0;
  let page = 0;

  while (heightLeft > 0) {
    if (page > 0) {
      pdf.addPage();
    }
    pdf.addImage(imgData, 'JPEG', 0, position, imgWidth, imgHeight);
    heightLeft -= pdfHeight;
    position -= pdfHeight;
    page++;
  }

  statusCallback?.('Descargando...');
  pdf.save(filename);
}
