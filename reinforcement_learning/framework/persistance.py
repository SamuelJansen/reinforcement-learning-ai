from pymongo import MongoClient
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne

from bson.objectid import ObjectId

from python_helper import Constant as c
from python_helper import ObjectHelper, RandomHelper, ReflectionHelper, log
from reinforcement_learning.framework.value import List, Dictionary, Tuple
from reinforcement_learning.ai.agent import Agent, AgentConstants
from reinforcement_learning.ai.action import Action
from reinforcement_learning.ai.environment import Environment
from reinforcement_learning.ai.episode import Episode
from reinforcement_learning.ai.history import History


class MongoDB:

    SET = '$set'
    EPISODES = 'Episodes'

    def __init__(self, agent: Agent, environment: Environment):
        self.agentKey = agent.getKey()
        self.agentClassName = agent.getType()
        self.databaseNameSufix: str = f'{environment.getKey()}'
        self.databaseName, self.client, self.collection = self.openMongoDbConnection(self.agentKey)
        self.episodeDatabaseName, self.episodeClient, self.episodeCollection = self.openMongoDbConnection(self.EPISODES)

    def openMongoDbConnection(self, sufix: str = c.BLANK):
        clientName = f'{self.agentClassName}'
        databaseName: str = f'{clientName}{self.databaseNameSufix}{sufix}'
        client: MongoClient = MongoClient(f'mongodb://localhost:27017/{clientName}')
        collection = ReflectionHelper.getAttributeOrMethod(client, clientName)
        return databaseName, client, collection

    def persistEpisode(self, episode: Episode, muteLogs: bool = False):
        agent = episode.agents.get(self.agentKey)
        if not muteLogs:
            log.debug(self.persistEpisode, f'Persiting {self.getAgentInfo(agent)} events')
        try:
            for event in episode.history:
                # log.prettyPython(self.persistEpisode, 'History', episode.history.asJson(), logLevel=log.DEBUG)
                self.episodeCollection[self.episodeDatabaseName].save({
                    '_id': episode.getId(),
                    'history': episode.history.asJson()
                })
        except Exception as exception:
            errorMessage = f'Not possible to persit {self.getAgentInfo(agent)} events'
            log.error(self.persistEpisode, errorMessage, exception)
            raise Exception(f'{errorMessage}. Cause: {str(exception)}')
        if not muteLogs:
            log.debug(self.persistEpisode, f'{self.getAgentInfo(agent)} events persisted')

    def persistActionTable(self, agent: Agent, muteLogs: bool = False):
        if not muteLogs:
            log.debug(self.persistActionTable, f'Persiting {self.getAgentInfo(agent)} action table')
        try:
            for stateHash, stateAction in agent.actionTable.items():
                self.collection[self.databaseName].save(
                    {
                        AgentConstants.ID_DB_KEY : stateAction[AgentConstants.ID],
                        AgentConstants.STATE_HASH_DB_KEY : stateHash,
                        AgentConstants.ACTIONS_DB_KEY : stateAction[AgentConstants.ACTIONS]
                    }
                )
            # for stateHash, stateAction in agent.getActionTable().items():
            #     self.collection[self.databaseName].update(
            #         {
            #             AgentConstants.ID_DB_KEY: stateAction[AgentConstants.ID]
            #         },
            #         {
            #             self.SET : {
            #                 AgentConstants.ID_DB_KEY: stateAction[AgentConstants.ID],
            #                 AgentConstants.STATE_HASH_DB_KEY,
            #                 AgentConstants.ACTIONS_DB_KEY : stateAction[AgentConstants.ACTIONS]
            #             }
            #         },
            #         upsert=True
            #     )
        except Exception as exception:
            errorMessage = f'Not possible to persit {self.getAgentInfo(agent)} action table'
            log.error(self.persistActionTable, errorMessage, exception)
            raise Exception(f'{errorMessage}. Cause: {str(exception)}')
        if not muteLogs:
            log.debug(self.persistActionTable, f'{self.getAgentInfo(agent)} action table persisted')

    def loadActionTable(self, agent: Agent, muteLogs: bool = False):
        if not muteLogs:
            log.debug(self.persistActionTable, f'Loading {self.getAgentInfo(agent)} action table')
        try:
            agent.setActionTable(
                {
                    dbEntry[AgentConstants.STATE_HASH_DB_KEY]: self.convertFromDbActionVisitsToAgentActionVisits(dbEntry) for dbEntry in self.collection[self.databaseName].find({})
                }
            )
        except Exception as exception:
            errorMessage = f'Not possible to load {self.getAgentInfo(agent)} action table'
            log.error(self.persistActionTable, errorMessage, exception)
            raise Exception(f'{errorMessage}. Cause: {str(exception)}')
        if not muteLogs:
            log.debug(self.persistActionTable, f'{self.getAgentInfo(agent)} action table loaded')

    def convertFromDbActionVisitsToAgentActionVisits(self, dbEntry):
        # print(dbEntry)
        return Dictionary({
            AgentConstants.ID: dbEntry[AgentConstants.ID_DB_KEY],
            AgentConstants.ACTIONS : List([
                self.convertFromDbActionVisitToAgentActionVisit(dbAction) for dbAction in dbEntry[AgentConstants.ACTIONS_DB_KEY]
            ])
        })

    def convertFromDbActionVisitToAgentActionVisit(self, dbActionVisit):
        return Dictionary({
            AgentConstants.ACTION : self.convertFromDbActionToAgentAction(dbActionVisit[AgentConstants.ACTION_DB_KEY]),
            AgentConstants.ACTION_VALUE : dbActionVisit[AgentConstants.ACTION_VALUE_DB_KEY],
            AgentConstants.ACTION_VISITS : dbActionVisit[AgentConstants.ACTION_VISITS_DB_KEY]
        })

    def convertFromDbActionToAgentAction(self, dbAction):
        if ObjectHelper.isNotCollection(dbAction) or (ObjectHelper.isCollection(dbAction) and ObjectHelper.isNotCollection(dbAction[0]) and 1 == len(dbAction)):
            return Action([tuple(dbAction)])
        return Action([tuple(*dbAction)])

    def getAgentInfo(self, agent: Agent):
        return f'{agent.getType()} {agent.getKey()} {self.databaseNameSufix}'


####################
EXAMPLE = '''
result = db.test.bulk_write([
    DeleteMany({}),  # Remove all documents from the previous example.
    InsertOne({'_id': 1}),
    InsertOne({'_id': 2}),
    InsertOne({'_id': 3}),
    UpdateOne({'_id': 1}, {'$set': {'foo': 'bar'}}),
    UpdateOne({'_id': 4}, {'$inc': {'j': 1}}, upsert=True),
    ReplaceOne({'j': 1}, {'j': 2})
])
'''
