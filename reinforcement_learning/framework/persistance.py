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
        self.agent = agent
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
        if not muteLogs:
            log.debug(self.persistEpisode, f'Persiting {self.getAgentInfo()} events')
        try:
            for event in episode.history:
                # log.prettyPython(self.persistEpisode, 'History', episode.history.asJson(), logLevel=log.DEBUG)
                self.episodeCollection[self.episodeDatabaseName].save({
                    '_id': episode.getId(),
                    'history': episode.history.asJson()
                })
        except Exception as exception:
            errorMessage = f'Not possible to persit {self.getAgentInfo()} events'
            log.error(self.persistEpisode, errorMessage, exception)
            raise Exception(f'{errorMessage}. Cause: {str(exception)}')
        if not muteLogs:
            log.debug(self.persistEpisode, f'{self.getAgentInfo()} events persisted')

    def persistActionTable(self, muteLogs: bool = False):
        if not muteLogs:
            log.debug(self.persistActionTable, f'Persiting {self.getAgentInfo()} action table')
        try:
            for stateHash, stateAction in self.agent.actionTable.items():
                self.collection[self.databaseName].save(
                    {
                        AgentConstants.ID_DB_KEY : stateAction[AgentConstants.ID],
                        AgentConstants.STATE_HASH_DB_KEY : stateHash,
                        AgentConstants.ACTIONS_DB_KEY : stateAction[AgentConstants.ACTIONS]
                    }
                )
            # for stateHash, stateAction in self.agent.getActionTable().items():
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
            errorMessage = f'Not possible to persit {self.getAgentInfo()} action table'
            log.error(self.persistActionTable, errorMessage, exception)
            raise Exception(f'{errorMessage}. Cause: {str(exception)}')
        if not muteLogs:
            log.debug(self.persistActionTable, f'{self.getAgentInfo()} action table persisted')

    def loadActionTable(self, muteLogs: bool = False):
        if not muteLogs:
            log.debug(self.persistActionTable, f'Loading {self.getAgentInfo()} action table')
        try:
            self.agent.setActionTable(
                {
                    dbEntry[AgentConstants.STATE_HASH_DB_KEY]: self.convertFromDbActionVisitsToAgentActionVisits(dbEntry) for dbEntry in self.collection[self.databaseName].find({})
                }
            )
        except Exception as exception:
            errorMessage = f'Not possible to load {self.getAgentInfo()} action table'
            log.error(self.persistActionTable, errorMessage, exception)
            raise Exception(f'{errorMessage}. Cause: {str(exception)}')
        if not muteLogs:
            log.debug(self.persistActionTable, f'{self.getAgentInfo()} action table loaded')

    def convertFromDbActionVisitsToAgentActionVisits(self, dbEntry):
        # print(dbEntry)
        return Dictionary({
            AgentConstants.ID: dbEntry[AgentConstants.ID_DB_KEY],
            AgentConstants.ACTIONS : List([
                temporarelyValidateCartPoleV1Actions(self.convertFromDbActionVisitToAgentActionVisit(dbAction) for dbAction in dbEntry[AgentConstants.ACTIONS_DB_KEY])
            ])
        })

    def temporarelyValidateCartPoleV1Actions(actions):
        if len(actions) > 2:
            error = {action.getHash(): action for action in actions}
            log.prettyPython(self._update, f'self.actions', error, logLevel=log.DEBUG)
            log.prettyPython(self._update, f'self.actions', error, logLevel=log.ERROR)
            exception = Exception('CartPole-V1 test exception')
            log.error(self.temporarelyValidateCartPoleV1Actions, 'Invalid actions', exception)
            raise exception
        return actions

    def convertFromDbActionVisitToAgentActionVisit(self, dbActionVisit):
        return Dictionary({
            AgentConstants.ACTION : self.convertFromDbActionToAgentAction(dbActionVisit[AgentConstants.ACTION_DB_KEY]),
            AgentConstants.ACTION_VALUE : dbActionVisit[AgentConstants.ACTION_VALUE_DB_KEY],
            AgentConstants.ACTION_VISITS : dbActionVisit[AgentConstants.ACTION_VISITS_DB_KEY]
        })

    def convertFromDbActionToAgentAction(self, dbAction):
        if ObjectHelper.isNotCollection(dbAction):
            return Action([tuple(dbAction)])
        elif ObjectHelper.isCollection(dbAction):
            if ObjectHelper.isNotCollection(dbAction[0]) and 1 == len(dbAction):
                return Action([tuple(dbAction)])
        print(dbAction)
        raise Exception('please verify it')
        return Action([tuple(*dbAction)])

    def getAgentInfo(self):
        return f'{self.agent.getType()} {self.agent.getKey()} {self.databaseNameSufix}'


def loadMongoDbAgents(environment: Environment, agents: dict) -> dict:
    mongoDbAgents: dict = {
        agentKey: {
            'agent': agent,
            'mongoDb': MongoDB(
                agent,
                environment
            )
        } for agentKey, agent in agents.items()
    }
    for agentKey, mongoDbAgent in mongoDbAgents.items():
        mongoDbAgent['mongoDb'].loadActionTable()
    return mongoDbAgents


def updateMongoDbAgents(mongoDbAgents: dict):
    for agentKey, mongoDbAgent in mongoDbAgents.items():
        mongoDbAgent['mongoDb'].persistActionTable()


def saveEpisode(mongoDbAgent: dict, episode: Episode, muteLogs: bool = False):
    mongoDbAgent['mongoDb'].persistEpisode(episode, muteLogs=muteLogs)

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
