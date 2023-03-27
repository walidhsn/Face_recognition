import mysql.connector

class connexion():
  def __init__(self,mydb):
    self.mydb = mydb
  def getcon(self):
    self.mydb = mysql.connector.connect(host="localhost", user="edukey", password="root", database="img_data")
    if self.mydb:
      print("\nConnecting to Oracle Data-base 100%.")
      return self.mydb
    else:
      print("Error :cannot Connect To the Data-Base")





