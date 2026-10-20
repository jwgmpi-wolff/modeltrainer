import fs from 'node:fs';
import path from 'node:path';
import PDFDocument from 'pdfkit';
import { Document, HeadingLevel, Packer, Paragraph, TextRun } from 'docx';

export function validateExport(body) {
  if (!body || !['pdf', 'docx'].includes(body.format) ||
      !Array.isArray(body.messages) || !body.messages.length || body.messages.length > 200 ||
      body.messages.some((message) => !message || !['user', 'assistant'].includes(message.role) ||
        typeof message.content !== 'string' || !message.content.trim() || message.content.length > 100000)) {
    throw new Error('Choose PDF or Word and provide 1-200 non-empty conversation messages.');
  }
  return {
    format: body.format,
    title: String(body.title || 'Model conversation').slice(0, 200),
    model: String(body.model || 'Unspecified model').slice(0, 200),
    messages: body.messages.map(({ role, content }) => ({ role, content })),
  };
}

export async function exportConversation(conversation) {
  const { title, model, messages, format } = conversation;
  if (format === 'docx') {
    const children = [
      new Paragraph({ text: title, heading: HeadingLevel.TITLE }),
      new Paragraph({ text: `Model: ${model}`, spacing: { after: 240 } }),
    ];
    for (const message of messages) {
      children.push(new Paragraph({
        text: message.role === 'user' ? 'Prompt' : 'Response',
        heading: HeadingLevel.HEADING_2,
      }));
      for (const line of message.content.split('\n')) {
        children.push(new Paragraph({
          children: [new TextRun(line)],
          spacing: { after: 100 },
        }));
      }
    }
    return Packer.toBuffer(new Document({ sections: [{ children }] }));
  }
  const doc = new PDFDocument({ margin: 50, bufferPages: true, info: { Title: title, Author: 'Local Model Lab' } });
  const chunks = [];
  const completed = new Promise((resolve, reject) => {
    doc.on('data', (chunk) => chunks.push(chunk));
    doc.on('end', () => resolve(Buffer.concat(chunks)));
    doc.on('error', reject);
  });
  const fonts = [
    path.join(process.env.WINDIR || 'C:\\Windows', 'Fonts', 'arial.ttf'),
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
  ];
  const font = fonts.find((candidate) => fs.existsSync(candidate));
  if (font) doc.font(font);
  doc.fontSize(22).fillColor('#15345c').text(title);
  doc.moveDown().fontSize(10).fillColor('#475569').text(`Model: ${model}`);
  for (const message of messages) {
    doc.moveDown().fontSize(13).fillColor('#15345c').text(message.role === 'user' ? 'Prompt' : 'Response');
    doc.moveDown(0.4).fontSize(11).fillColor('#111827').text(message.content, { lineGap: 4 });
  }
  const pages = doc.bufferedPageRange();
  for (let index = pages.start; index < pages.start + pages.count; index += 1) {
    doc.switchToPage(index);
    doc.fontSize(9).fillColor('#475569').text(`Local Model Lab | ${index + 1} / ${pages.count}`, 50, doc.page.height - 40, {
      lineBreak: false,
    });
  }
  doc.end();
  return completed;
}
