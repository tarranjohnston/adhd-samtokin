# Requires the installation of the following libraries:
# googletrans
import random
import os
import csv
import asyncio
from googletrans import Translator
from example_emails import example_email
from data_for_email_construction import actual_categories, info_req_categories, info_req_subjects, info_req_details, web_shop_categories, web_shop_order_details, web_shop_subjects, course_confirmation_categories, course_confirmation_details, course_confirmation_subjects, intros, closings, usernames


# Generate random subjects with varying formats
async def generate_subject(category, rand_trans):
    categories, subjects, details = get_category(category)
    sub_category = random.choice(categories)
    
    # Select a random subject based on the input category
    subject = random.choice(subjects.get(sub_category))
    
    if rand_trans:
        subject = await translate_to_icelandic(subject)
    return subject

# Generate random senders
def generate_sender(category):
    # Automated Sender
    if category == "Course Confirmation":
        return "noreply@virk.is"
    
    # Person sender
    username = random.choice(usernames)
    domain = random.choice(["gmail.com", "yahoo.com", "outlook.com"])
    return f"{username}@{domain}"

    
# Function to generate a category detail by selecting one sentence from each part
def generate_category_detail(category):
    categories, subjects, details = get_category(category)
    sub_category = random.choice(categories)
    
    # Select one sentence from each part
    part1 = random.choice(details[sub_category]["part1"])
    part2 = random.choice(details[sub_category]["part2"])
    part3 = random.choice(details[sub_category]["part3"])
    
    # Combine the sentences into one detail string
    detail = f"{part1} {part2} {part3}"
    return detail

# Generate body text tailored to the category with a random detail
async def generate_body(category, rand_trans):
    intro = random.choice(intros)
    detail = generate_category_detail(category)
    closing = random.choice(closings)
    body_text = f"{intro} {detail} {closing}"
    if rand_trans:
        body_text =  await translate_to_icelandic(body_text)
    return body_text

# Function to translate text into Icelandic (using googletrans API for simplicity)
async def translate_to_icelandic(text):
    translator = Translator()
    translated = await translator.translate(text, src='en', dest='is')
    return translated.text

# Function to fetch the related subcategory from the category
def get_category(category):
    if category == "Information Request":
        return info_req_categories, info_req_subjects, info_req_details
    elif category == "Web Shop Order":
        return web_shop_categories, web_shop_subjects, web_shop_order_details
    elif category == "Course Confirmation":
        return course_confirmation_categories, course_confirmation_subjects, course_confirmation_details




async def main():
    emails = []
    counter = 0
    for _ in range(2000):
        counter += 1
        
        # Chance for random translation to icelandic
        rand_trans = False
        if random.random() < 0.5:
            rand_trans = True
            print(f"Translating email {counter}")

        # Chance for constructing emails, or choosing pre-constructed emails
        if random.random() < 0.1:
            emails.append(random.choice(example_email))
        else:
            category = random.choice(actual_categories)
            sender = generate_sender(category)
            subject = await generate_subject(category, rand_trans)
            body = await generate_body(category, rand_trans)
            emails.append({
                "subject": subject,
                "sender": sender,
                "body": body,
                "category": category
            })

    # Get the current working directory
    current_directory = os.getcwd()

    # Define the file path in the current directory
    file_path = os.path.join(current_directory, 'charity_emails_v10.csv')
    print(f"Directory: {file_path}")

    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["subject", "sender", "body", "category"])
        writer.writeheader()
        writer.writerows(emails)
        print("Generating CSV")

asyncio.run(main())
