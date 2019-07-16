CREATE DATABASE TennisML
GO
USE TennisML
GO
CREATE SCHEMA Stat
GO

CREATE TABLE Stat.Players (
	PlayerID INT PRIMARY KEY,
	Name VARCHAR(255),
	Birthdate DATE,
	Country VARCHAR(3),
	Prize INT,
	ATPPoints INT,
	ATPRank INT
)

CREATE TABLE Stat.Surface (
	SurfaceID INT PRIMARY KEY,
	Name VARCHAR(255)
)

CREATE TABLE Stat.Tournaments (
	TournamentID INT PRIMARY KEY,
	Name VARCHAR(255),
	SurfaceID INT,
	Date DATE,
	RankID INT,
	Country VARCHAR(3)
)

CREATE TABLE Stat.Matches (
	MatchID INT PRIMARY KEY,
	PlayerID_1 INT,
	PlayerID_2 INT,
	TournamentID INT,
	Date DATE,
	Result VARCHAR(255),
	Odds_Marathonbet FLOAT,
	Odds_Pinnacle FLOAT,
	FirstServePercentage_1 FLOAT,
	FirstServePercentage_2 FLOAT,
	Aces_1 INT,
	Aces_2 INT,
	DoubleFaults_1 INT,
	DoubleFaults_2 INT,
	UnforcedErrors_1 INT,
	UnforcedErrors_2 INT,
	WonFirstServePercentage_1 FLOAT,
	WonFirstServePercentage_2 FLOAT,
	WonSecondServePercentage_1 FLOAT,
	WonSecondServePercentage_2 FLOAT,
	WonReturnPercentage_1 FLOAT,
	WonReturnPercentage_2 FLOAT,
	Winners_1 INT,
	Winners_2 INT,
	BreakPointsWon_1 INT,
	BreakPointsWon_2 INT,
	BreakPointsTotal_1 INT,
	BreakPointsTotal_2 INT,
	NetApproachesWon_1 INT,
	NetApproachesWon_2 INT,
	NetApproachesTotal_1 INT,
	NetApproachesTotal_2 INT,
	TotalPointsWon_1 INT,
	TotalPointsWon_2 INT,
	FastestServe_1 INT,
	FastestServe_2 INT,
	AverageFirstServeSpeed_1 INT,
	AverageFirstServeSpeed_2 INT,
	AverageSecondServeSpeed_1 INT,
	AverageSecondServeSpeed_2 INT
)

