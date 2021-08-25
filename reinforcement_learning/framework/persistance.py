from pymongo import MongoClient
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne

from bson.objectid import ObjectId

from python_helper import ObjectHelper, RandomHelper, ReflectionHelper, log
from reinforcement_learning.framework.value import List, Dictionary, Tuple
from reinforcement_learning.ai.agent import Agent, AgentConstants
from reinforcement_learning.ai.action import Action


class MongoDB:

    SET = '$set'

    def __init__(self, agentClassName: str, databaseNameSufix: str):
        self.agentClassName = agentClassName
        self.databaseNameSufix: str = databaseNameSufix
        self.openMongoDbConnection(self.databaseNameSufix)

    def openMongoDbConnection(self, sufix: str):
        self.databaseName: str = f'{self.agentClassName}{sufix}'
        self.client: MongoClient = MongoClient(f'mongodb://localhost:27017/{self.databaseName}')
        self.collection = ReflectionHelper.getAttributeOrMethod(self.client, self.databaseName)

    def persistActionTable(self, agent: Agent):
        log.debug(self.persistActionTable, f'Persiting {ReflectionHelper.getClassName(agent)} {agent.getKey()} action table')
        try:
            for stateHash, stateAction in agent.getActionTable().items():
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
            log.error(self.persistActionTable, f'Not possible to persit {ReflectionHelper.getClassName(agent)} {agent.getKey()} action table', exception)
            return None
        log.debug(self.persistActionTable, f'{ReflectionHelper.getClassName(agent)} {agent.getKey()} action table persisted')

    def loadActionTable(self, agent: Agent):
        log.debug(self.persistActionTable, f'Loading {ReflectionHelper.getClassName(agent)} {agent.getKey()} action table')
        try:
            agent.setActionTable(
                Dictionary({
                    element[AgentConstants.STATE_HASH_DB_KEY]: {
                        AgentConstants.ID: element[AgentConstants.ID_DB_KEY],
                        AgentConstants.ACTIONS : List([
                            {
                                AgentConstants.ACTION : self.convertFromDbActionToAgentAction(action[AgentConstants.ACTION_DB_KEY]),
                                AgentConstants.ACTION_VALUE : action[AgentConstants.ACTION_VALUE_DB_KEY],
                                AgentConstants.ACTION_VISITS : action[AgentConstants.ACTION_VISITS_DB_KEY]
                            } for action in element[AgentConstants.ACTIONS_DB_KEY]
                        ])
                    } for element in self.collection[self.databaseName].find({})
                })
            )
        except Exception as exception:
            log.error(self.persistActionTable, f'Not possible to loat {ReflectionHelper.getClassName(agent)} {agent.getKey()} action table', exception)
            return None
        log.debug(self.persistActionTable, f'{ReflectionHelper.getClassName(agent)} {agent.getKey()} action table loaded')

    def convertFromDbActionToAgentAction(self, action):
        if ObjectHelper.isNotCollection(action) or (ObjectHelper.isCollection(action) and 1 == len(action)):
            return Action([tuple(action)])
        return Action([tuple(*action)])



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
