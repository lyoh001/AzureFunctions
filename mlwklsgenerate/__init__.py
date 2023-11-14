import logging
import os
import shutil
from io import BytesIO

import azure.functions as func
import requests
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

max_width = 492
input_pdf_path = "input_pdf.pdf"
output_pdf_path = "output_pdf_with_text.pdf"
korean_font_path = "/tmp/gulim.ttf"
korean_font_url = "https://github.com/lyoh001/AzureStaticWebApps/raw/main/wkls/font.ttf"


def download_korean_font(font_url, font_path):
    response = requests.get(font_url)
    with open(font_path, "wb") as font_file:
        font_file.write(response.content)


def load_template(id: str) -> str:
    src_path = "./mlwklsgenerate/template"
    dst_path = os.path.join("/tmp", id)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)
        if os.path.isdir(item_path):
            shutil.copytree(item_path, os.path.join(dst_path, item))
        else:
            shutil.copy2(item_path, dst_path)
    return dst_path


def generate_chart(score):
    if score == "0":
        text_x = 167
    elif score == "1":
        text_x = 253
    elif score == "2":
        text_x = 337
    elif score == "3":
        text_x = 423
    else:
        text_x = 509
    return text_x, "X"


def add_text_to_pdf(input_path, output_path, text_content, x, y, max_width, font_size):
    pdf_reader = PdfReader(input_path)
    pdf_writer = PdfWriter()

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        packet = BytesIO()
        c = canvas.Canvas(packet)
        c.setFont("gulim", font_size)
        lines = []
        current_line = ""
        text_lines = text_content.split("\n")
        for line in text_lines:
            words = line.split()
            for word in words:
                test_line = current_line + word + " "
                width = c.stringWidth(test_line, "gulim", font_size)
                if width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            lines.append(current_line.strip())
            current_line = ""

        text_x = x
        text_y = y

        for line in lines:
            c.drawString(text_x, text_y, line)
            text_y -= font_size * 1.2
        c.save()
        packet.seek(0)
        modified_page = PdfReader(packet).pages[0]
        page.merge_page(modified_page)
        pdf_writer.add_page(page)

    with open(output_path, "wb") as output_file:
        pdf_writer.write(output_file)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting Generate function*******")
    download_korean_font(korean_font_url, korean_font_path)
    pdfmetrics.registerFont(TTFont("gulim", korean_font_path))
    try:
        user_input = req.get_json()
        student_name = user_input["studentName"]
        grade = int(user_input["grade"])
        attendance = user_input["attendance"]
        behaviour = user_input["behaviour"]
        effort = user_input["effort"]
        communucation_skills = "\n".join(
            f"- {skill.split(']')[1]}" for skill in user_input["communicationSkills"]
        )
        understanding_skills = "\n".join(
            f"- {skill.split(']')[1]}" for skill in user_input["understandingSkills"]
        )
        overall_comment = user_input["overallComment"]
        path = load_template(student_name)
        text = student_name
        text_x = 155
        text_y = 704
        add_text_to_pdf(
            os.path.join(path, input_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            11,
        )

        text = f"{'Year' if grade else ''} {grade if grade else 'Prep'}"
        text_x = 390
        text_y = 704
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            11,
        )

        text_x = 57
        text_y = 625
        text_y = 652
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            communucation_skills,
            text_x,
            text_y,
            max_width,
            9,
        )

        text_x = 57
        text_y = 480
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            understanding_skills,
            text_x,
            text_y,
            max_width,
            9,
        )

        text_x = 57
        text_y = 305
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            overall_comment,
            text_x,
            text_y,
            max_width,
            9,
        )

        text_x, text = generate_chart(attendance)
        text_y = 119
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            10,
        )

        text_x, text = generate_chart(behaviour)
        text_y = 102
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            10,
        )

        text_x, text = generate_chart(effort)
        text_y = 84
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            10,
        )

        if os.path.exists(os.path.join(path, output_pdf_path)):
            with open(os.path.join(path, output_pdf_path), "rb") as file:
                file_content = file.read()
            return func.HttpResponse(
                body=file_content,
                headers={
                    "Content-Disposition": "attachment; filename=report.pdf",
                    "Content-Type": "application/pdf",
                },
                status_code=200,
            )
        else:
            return func.HttpResponse("Error generating the report", status_code=500)

    except ValueError:
        return func.HttpResponse(
            "Invalid input, please provide a valid user_input.", status_code=400
        )

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
