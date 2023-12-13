import mysql.connector

TOSS_TABLE = "toss"
STATS_TABLE = "stats"


# open connection to DB
def connect(user='root', password='m7porHiFaNET4tja', host='10.5.32.16', database='dicetoss'):
    cnx = mysql.connector.connect(user=user, password=password,
                                  host=host,
                                  database=database)

    return cnx


# close connection to DB
def close_connection(cnx):
    cnx.close()

def init_stats(cnx):

    delete = "DELETE FROM " + STATS_TABLE
    insert = "INSERT INTO " + STATS_TABLE + " (toss, count) VALUES (%s, %s)"

    cursor = cnx.cursor()

    try:
        cursor.execute(delete)
        for char in ('A','B','C','D','E','F','X'):
            cursor.execute(insert, (char, 0))
    # just in case
    except mysql.connector.Error as e:
        print(e)

    cnx.commit()
    cursor.close()

    return

def delete_tosses(cnx):
    delete = "DELETE FROM " + TOSS_TABLE
    cursor = cnx.cursor()

    try:
        cursor.execute(delete)
    # just in case
    except mysql.connector.Error as e:
        print(e)

    cnx.commit()
    cursor.close()

    return

# insert toss and update stats
def insert_toss(cnx, toss):

    cursor = cnx.cursor()

    # insert toss
    sql_toss = "INSERT INTO " + TOSS_TABLE + " (toss, timestamp, filename) VALUES (%s, %s, %s)"
    values_toss = (toss['toss'], toss['time'], toss['filename'])

    # update stats
    update = "UPDATE " + STATS_TABLE + " SET count = count + 1 WHERE toss = '" + toss['toss'] + "'"

    try:
        cursor.execute(sql_toss, values_toss)
        cursor.execute(update)
    # just in case
    except mysql.connector.Error as e:
        print(e)

    cnx.commit()
    cursor.close()

    return

def main():
    cnx = connect()
    delete_tosses(cnx)
    init_stats(cnx)
    toss = {
            'toss':'A',
            'time':'123456789',
            "filename":'TBA'
            }
    #insert_toss(cnx, toss)
    close_connection(cnx)

if __name__ == "__main__":
    main()
