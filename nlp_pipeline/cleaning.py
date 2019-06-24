import requests
from bs4 import BeautifulSoup

# fetch web page
r = requests.get("https://www.udacity.com/courses/all")
#print(r.text)

# use BeautifulSoup to remove HTML tags
soup = BeautifulSoup(r.text, "lxml")
for script in soup(["script", "style"]):
    script.decompose()
#print(soup.get_text())

# find all course summaries
summaries = soup.find_all("div", {"class":"course-summary-card"})
print('Number of Courses:', len(summaries))

# print the first summary in summaries
#print(summaries[0].prettify())

# Extract course title
#summaries[0].select_one("h3").get_text().strip()

# Extract school
#summaries[0].select_one("h4").get_text().strip()

# Collect names and schools of ALL course listings
courses = []
for summary in summaries:
    title = summary.select_one("h3").get_text().strip()
    school = summary.select_one("h4").get_text().strip()
    courses.append((title, school))

# display results
print(len(courses), "course summaries found. Sample:")
print(courses[:20])