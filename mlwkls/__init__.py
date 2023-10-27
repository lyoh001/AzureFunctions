import logging
import os
import shutil
from io import BytesIO

import azure.functions as func
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

max_width = 480
input_pdf_path = "input_pdf.pdf"
output_pdf_path = "output_pdf_with_text.pdf"


def load_template(id: str) -> str:
    src_path = "./mlwkls/template"
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


def get_rating(rating):
    return {
        "0": "Needs Attention",
        "1": "Satisfactory",
        "2": "Very Good",
        "3": "Excellent",
    }.get(rating, "Unknown")


def generate_chart(previous_semester, current_semester):
    difference_semester = current_semester - previous_semester
    score = ""
    if difference_semester > 0:
        score = f"X{'-------------------------' * abs(difference_semester)}O"
    elif difference_semester < 0:
        score = f"O{'-------------------------' * abs(difference_semester)}X"
    else:
        score = "X-O"

    if min(previous_semester, current_semester) == 0:
        text_x = 165
    elif min(previous_semester, current_semester) == 1:
        text_x = 250
    elif min(previous_semester, current_semester) == 2:
        text_x = 335
    elif min(previous_semester, current_semester) == 3:
        text_x = 420
    else:
        text_x = 505
    return text_x, score


def add_text_to_pdf(input_path, output_path, text_content, x, y, max_width, font_size):
    pdf_reader = PdfReader(input_path)
    pdf_writer = PdfWriter()

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        packet = BytesIO()
        c = canvas.Canvas(packet)
        c.setFont("Helvetica", font_size)
        lines = []
        current_line = ""
        text_lines = text_content.split("\n")
        for line in text_lines:
            words = line.split()
            for word in words:
                test_line = current_line + word + " "
                width = c.stringWidth(test_line, "Helvetica", font_size)
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
    logging.info("*******Starting Report function*******")
    try:
        user_input = req.get_json()
        student_name = user_input["studentName"]
        grade = user_input["grade"]
        attendance = user_input["attendance"]
        effort = user_input["effort"]
        behavior = user_input["behavior"]
        listening_previous_semester = int(user_input["listeningPreviousSemester"])
        listening_current_semester = int(user_input["listeningCurrentSemester"])
        listening_skills = user_input["listeningSkills"]
        reading_previous_semester = int(user_input["readingPreviousSemester"])
        reading_current_semester = int(user_input["readingCurrentSemester"])
        reading_skills = user_input["readingSkills"]
        speaking_previous_semester = int(user_input["speakingPreviousSemester"])
        speaking_current_semester = int(user_input["speakingCurrentSemester"])
        speaking_skills = user_input["speakingSkills"]
        writing_previous_semester = int(user_input["writingPreviousSemester"])
        writing_current_semester = int(user_input["writingCurrentSemester"])
        writing_skills = user_input["writingSkills"]
        overall_comment = user_input["overallComment"]

        path = load_template(student_name)

        text = f"{student_name} / Year {grade}"
        text_x = 160
        text_y = 702
        add_text_to_pdf(
            os.path.join(path, input_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            11,
        )

        text = get_rating(attendance)
        text_x = 160
        text_y = 678
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            11,
        )

        text = get_rating(effort)
        text_x = 420
        text_y = 702
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            11,
        )

        text = get_rating(behavior)
        text_x = 420
        text_y = 678
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            11,
        )

        text_x, text = generate_chart(
            listening_previous_semester, listening_current_semester
        )
        text_y = 183
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            10,
        )

        text_x, text = generate_chart(
            reading_previous_semester, reading_current_semester
        )
        text_y = 161
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            10,
        )

        text_x, text = generate_chart(
            speaking_previous_semester, speaking_current_semester
        )
        text_y = 141
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            10,
        )

        text_x, text = generate_chart(
            writing_previous_semester, writing_current_semester
        )
        text_y = 120
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            text,
            text_x,
            text_y,
            max_width,
            10,
        )

        text_x = 60
        text_y = 625
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            "\n".join(listening_skills),
            text_x,
            text_y,
            max_width,
            9,
        )

        text_x = 60
        text_y = 538
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            "\n".join(reading_skills),
            text_x,
            text_y,
            max_width,
            9,
        )

        text_x = 60
        text_y = 450
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            "\n".join(speaking_skills),
            text_x,
            text_y,
            max_width,
            9,
        )

        text_x = 60
        text_y = 360
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            "\n".join(writing_skills),
            text_x,
            text_y,
            max_width,
            9,
        )

        text_x = 60
        text_y = 276
        add_text_to_pdf(
            os.path.join(path, output_pdf_path),
            os.path.join(path, output_pdf_path),
            overall_comment,
            text_x,
            text_y,
            max_width,
            9,
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
