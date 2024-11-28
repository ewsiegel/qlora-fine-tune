import csv
import json
from random import random as _random
import requests
from string import ascii_letters as _ascii_letters
from string import digits as _digits
import time
from time import time as _time


def request(method, session, data=None, nid=None, nid_key='nid',
                api_type="logic", return_response=False):
    """Get data from arbitrary Piazza API endpoint `method` in network `nid`

    :type  method: str
    :param method: An internal Piazza API method name like `content.get`
        or `network.get_users`
    :type  data: dict
    :param data: Key-value data to pass to Piazza in the request
    :type  nid: str
    :param nid: This is the ID of the network to which the request
        should be made. This is optional and only to override the
        existing `network_id` entered when creating the class
    :type  nid_key: str
    :param nid_key: Name expected by Piazza for `nid` when making request.
        (Usually and by default "nid", but sometimes "id" is expected)
    :returns: Python object containing returned data
    :type return_response: bool
    :param return_response: If set, returns whole :class:`requests.Response`
        object rather than just the response body
    """

    if data is None:
        data = {}

    headers = {}
    if "session_id" in session.cookies:
        headers["CSRF-Token"] = session.cookies["session_id"]

    base_api_urls = {
            "logic": "https://piazza.com/logic/api",
            "main": "https://piazza.com/main/api",
        }
    # Adding a nonce to the request
    endpoint = base_api_urls[api_type]

    def _int2base(x, base):
        """
        Converts an integer from base 10 to some arbitrary numerical base,
        and return a string representing the number in the new base (using
        letters to extend the numerical digits).

        :type     x: int
        :param    x: The integer to convert
        :type  base: int
        :param base: The base to convert the integer to
        :rtype: str
        :returns: String representing the number in the new base
        """
        
        if base > len(_exradix_digits):
            raise ValueError(
                "Base is too large: The defined digit set only allows for "
                "bases smaller than " + len(_exradix_digits) + "."
            )

        if x > 0:
            sign = 1
        elif x == 0:
            return _exradix_digits[0]
        else:
            sign = -1

        x *= sign
        digits = []

        while x:
            digits.append(
                _exradix_digits[int(x % base)])
            x = int(x / base)

        if sign < 0:
            digits.append('-')

        digits.reverse()

        return ''.join(digits)
    
    _exradix_digits = _digits + _ascii_letters

    def nonce():
        """
        Returns a new nonce to be used with the Piazza API.
        """
        nonce_part1 = _int2base(int(_time()*1000), 36) 
        nonce_part2 = _int2base(round(_random()*1679616), 36)
        return "{}{}".format(nonce_part1, nonce_part2)
    

    if api_type == "logic":
        endpoint += "?method={}&aid={}".format(
            method,
            nonce()
        )

    response = session.post(
        endpoint,
        data=json.dumps({
            "method": method,
            "params": dict({nid_key: nid}, **data)
        }),
        headers=headers
    )
    return response if return_response else response.json()

def get_feed(nid, session, limit=100, offset=0):
    sort="updated"
    r = request(
            method="network.get_my_feed",
            session=session,
            nid=nid,
            data=dict(
                limit=limit,
                offset=offset,
                sort=sort
            )
        )
    return _handle_error(r, "Could not get the feed")

def _handle_error(result, err_msg):
        """Check result for error

        :type result: dict
        :param result: response body
        :type err_msg: str
        :param err_msg: The message given to the :class:`RequestError` instance
            raised
        :returns: Actual result from result
        :raises RequestError: If result has error
        """
        if result.get(u'error'):
            print(err_msg)
        else:
            return result.get(u'result')

def get_post(cid, nid, session):
    r = request(
            method="content.get",
            session=session,
            data={"cid": cid, "student_view": None},
            nid=nid
        )
    return _handle_error(r, "Could not get post {}.".format(cid))

def get_all_posts(nid, session, limit=None, sleep=0):
    feed = get_feed(nid=nid, session=session, limit=999999, offset=0)
    cids = [post['id'] for post in feed["feed"]]
    sleep = 0
    for cid in cids:
        time.sleep(sleep)
        yield get_post(cid, nid, session)

# Function to process a single post
def process_post(post, csv_file):
    # Extract data
    created = post['created']
    folders = ', '.join(post['folders'])
    id_number = post['nr']
    subject = post['history'][0]['subject']
    question = post['history'][0]['content']

    # Filter instructor and student answers
    instructor_answers = [
        child['history'][0]['content']
        for child in post['children']
        if child['type'] == 'i_answer'
    ]
    student_answers = [
        child['history'][0]['content']
        for child in post['children']
        if child['type'] == 's_answer'
    ]

    # Join answers as strings
    instructor_answers_text = '; '.join(instructor_answers) if instructor_answers else 'No instructor answer'
    student_answers_text = '; '.join(student_answers) if student_answers else 'No student answer'

    # Append data to CSV
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([created, folders, id_number, subject, question, instructor_answers_text, student_answers_text])



def main():
    session = requests.Session()

    response = session.get('https://piazza.com/main/csrf_token')

    # Make sure a CSRF token was retrieved, otherwise bail
    if response.text.upper().find('CSRF_TOKEN') == -1:
        print("ERROR with CSRF_TOKEN")
        return
        # raise AuthenticationError("Could not get CSRF token")

    # Remove double quotes and semicolon (ASCI 34 & 59) from response string.
    # Then split the string on "=" to parse out the actual CSRF token
    csrf_token = response.text.translate({34: None, 59: None}).split("=")[1]

    # PIAZZA LOG IN CREDENTIALS, FILL IN 
    email = ''
    password = ''

    # Log in using credentials and CSRF token and store cookie in session
    response = session.post(
        "https://piazza.com/class",
        data={
            "from": "/signup",
            "email": email,
            "password": password,
            "remember": "on",
            "csrf_token": csrf_token,
        },
    )

    # If non-successful http response, bail
    if response.status_code != 200:
        print("Response status not 200")
        # raise AuthenticationError(f"Could not authenticate.\n{response.text}")

    pos = response.text.upper().find('VAR ERROR_MSG')
    if pos != -1:
        print("ERROR")

    # CLASS ID
    nid = 'm05fat1q3i87bn'
    # CSV FILE NAME
    csv_file = 'post_data.csv'

    # Initialize CSV with headers
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Created Date', 'Folders', 'ID', 'Subject', 'Question', 'Instructor Answers', 'Student Answers'])

    # Loop through all posts and process them
    for _, post in enumerate(get_all_posts(nid, session)):
        if post == None:
            print("ERROR POST is NONE")
            continue
        else:
            process_post(post, csv_file)
            print("Processed post")

    print(f'All posts processed and saved to {csv_file}')


if __name__ == "__main__":
    main()

