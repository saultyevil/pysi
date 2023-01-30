import sqlalchemy
from sqlalchemy import Column, Float, Integer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Photon(Base):
    """Photon object for SQL database."""
    __tablename__ = "Photons"
    id = Column(Integer, primary_key=True, autoincrement=True)
    np = Column(Integer)
    freq = Column(Float)
    wavelength = Column(Float)
    weight = Column(Float)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    scat = Column(Integer)
    rscat = Column(Integer)
    delay = Column(Integer)
    spec = Column(Integer)
    orig = Column(Integer)
    res = Column(Integer)
    lineres = Column(Integer)

    def __repr__(self):
        return str(self.id)


def get_photon_db(root, cd=".", dd_dev=False, commitfreq=1000000):
    """Create or open a database to store the delay_dump file in an easier to
    query data structure.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    cd: str [optional]
        The directory containing the simulation.
    dd_dev: bool [optional]
        Expect the delay_dump file to be in the format used in the main Python
        repository.
    commitfreq: int
        The frequency to which commit the database and avoid out-of-memory
        errors. If this number is too low, database creation will take a long
        time.

    Returns
    -------
    engine:
        The SQLalchemy engine.
    session:
        The SQLalchemy session.
    """

    engine = sqlalchemy.create_engine("sqlite:///{}.db".format(root))
    engine_session = sessionmaker(bind=engine)
    session = engine_session()

    if dd_dev:
        column_names = [
            "Freq", "Lambda", "Weight", "LastX", "LastY", "LastZ", "Scat", "RScat", "Delay", "Spec", "Orig", "Res"
        ]
    else:
        column_names = [
            "Np", "Freq", "Lambda", "Weight", "LastX", "LastY", "LastZ", "Scat", "RScat", "Delay", "Spec", "Orig",
            "Res", "LineRes"
        ]
    n_columns = len(column_names)

    try:
        session.query(Photon.weight).first()
    except SQLAlchemyError:
        print("{}.db does not exist, so creating now".format(root))
        with open(cd + "/" + root + ".delay_dump", "r") as infile:
            nadd = 0
            Base.metadata.create_all(engine)
            for n, line in enumerate(infile):
                if line.startswith("#"):
                    continue
                try:
                    values = [float(i) for i in line.split()]
                except ValueError:
                    print("Line {} has values which cannot be converted into a number".format(n))
                    continue
                if len(values) != n_columns:
                    print("Line {} has unknown format with {} columns:\n{}".format(n, len(values), line))
                    continue
                if dd_dev:
                    session.add(
                        Photon(np=int(n),
                               freq=values[0],
                               wavelength=values[1],
                               weight=values[2],
                               x=values[3],
                               y=values[4],
                               z=values[5],
                               scat=int(values[6]),
                               rscat=int(values[7]),
                               delay=int(values[8]),
                               spec=int(values[9]),
                               orig=int(values[10]),
                               res=int(values[11]),
                               lineres=int(values[11])))
                else:
                    session.add(
                        Photon(np=int(values[0]),
                               freq=values[1],
                               wavelength=values[2],
                               weight=values[3],
                               x=values[4],
                               y=values[5],
                               z=values[6],
                               scat=int(values[7]),
                               rscat=int(values[8]),
                               delay=int(values[9]),
                               spec=int(values[10]),
                               orig=int(values[11]),
                               res=int(values[12]),
                               lineres=int(values[13])))

                nadd += 1
                if nadd > commitfreq:
                    session.commit()
                    nadd = 0

        session.commit()

    return engine, session
