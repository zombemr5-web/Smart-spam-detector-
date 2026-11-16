"""Generates a bilingual (EN/AR) sample dataset: data/spam.csv

NOTE: this is a SYNTHETIC demo dataset built from templates so the project
runs out of the box. For real results, replace data/spam.csv with a real
dataset (e.g. the UCI SMS Spam Collection) using columns: label,text
where label is 'spam' or 'ham'.
"""
import csv
import random

random.seed(42)

spam_en = [
    "Congratulations! You won a {prize}. Click {link} to claim now",
    "URGENT: your account is suspended. Verify at {link} immediately",
    "Free {prize} waiting for you! Reply YES to claim today only",
    "You have been selected for a cash reward of {amount}. Call now",
    "Limited offer!!! Get {pct}% off, buy now at {link}",
    "Your bank card is blocked. Confirm your password here {link}",
    "Earn {amount} per day from home, no experience needed. Join {link}",
    "Winner! Claim your free {prize} before midnight {link}",
]
ham_en = [
    "Hey, are we still meeting at {time} for the project?",
    "Can you send me the lecture notes for {subject}?",
    "I will be late, the bus is stuck in traffic",
    "Happy birthday! Hope you have a great day with the family",
    "Don't forget the {subject} exam is tomorrow morning",
    "Thanks for helping me with the assignment yesterday",
    "Dinner at my place on Friday? Let me know",
    "Please review my code and tell me what you think",
]
spam_ar = [
    "مبروك! ربحت {prize_ar}. اضغط على {link} لاستلام الجائزة الان",
    "عاجل: حسابك البنكي موقوف. ادخل على {link} لتأكيد بياناتك",
    "عرض محدود!!! خصم {pct}% على كل المنتجات اشتري الان {link}",
    "ربح {amount} دولار يوميا من البيت بدون خبرة. سجل هنا {link}",
    "تم اختيارك للفوز بجائزة نقدية. ارسل رقم بطاقتك للتأكيد",
    "هديه مجانيه لك فقط اليوم! رد بكلمة نعم للحصول عليها",
]
ham_ar = [
    "هلا، شكو ماكو؟ نلتقي الساعه {time} بالكلية؟",
    "تكدر ترسللي محاضرة {subject_ar} اذا ممكن؟",
    "راح اتاخر شوية، السيارة عالقة بالزحمة",
    "كل عام وانت بخير، الله يبارك لك بعمرك",
    "لا تنسى امتحان {subject_ar} باجر الصبح",
    "شكرا على مساعدتك بالواجب امس",
    "العشاء عندنا يوم الجمعه، اذا تحب تعال",
    "شوف الكود مالتي وكلي رايك بيه",
]

fill = {
    "prize": ["iPhone 15", "gift card", "new laptop", "free vacation", "$1000 voucher"],
    "prize_ar": ["ايفون جديد", "سيارة", "بطاقة هدايا", "رحلة مجانية"],
    "link": ["http://bit.ly/x7Kd2", "www.win-big-now.com", "http://secure-login.xyz/verify"],
    "amount": ["500", "1000", "2500", "$300"],
    "pct": ["50", "70", "90"],
    "time": ["5", "6", "10 AM", "3 PM"],
    "subject": ["math", "physics", "programming", "databases"],
    "subject_ar": ["الرياضيات", "الفيزياء", "البرمجة", "قواعد البيانات"],
}


def render(template):
    out = template
    for key, options in fill.items():
        out = out.replace("{" + key + "}", random.choice(options))
    return out


rows = []
for templates, label in [(spam_en, "spam"), (ham_en, "ham"), (spam_ar, "spam"), (ham_ar, "ham")]:
    for t in templates:
        for _ in range(12):
            rows.append((label, render(t)))

random.shuffle(rows)
with open("data/spam.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["label", "text"])
    w.writerows(rows)
print(f"Wrote {len(rows)} rows")
