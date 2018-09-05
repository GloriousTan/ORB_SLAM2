/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
}

/**
 * @brief Bag of Words Representation
 *
 * 计算mBowVec，并且将描述子分散在第4层上，即mFeatVec记录了属于第i个node的ni个描述子
 * @see ProcessNewKeyFrame()
 */
void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;
    //     |Rcw tcw|      |Rwc Ow|
    // Tcw=| 0   1 |  Twc=| 0   1|  
    // Ow=-Rwc*tcw表示左目相机中心坐标,即世界参考系(第一帧)原点(相机光心)在当前帧参考系(相机坐标系)中的坐标,等价于twc

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));

    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    //        |mHalfBaseline|
    // center=|     0       |为相机坐标系(左目)下立体相机中心的坐标
    //        |     0       |
    //        |     1       |
    Cw = Twc*center;
    // Cw表示世界坐标系下,左目相机中心到立体相机中心的向量
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

/**
 * @brief 为关键帧之间添加连接
 * 
 * 更新了mConnectedKeyFrameWeights中pKF对应的weight.
 * @param pKF    关键帧
 * @param weight 权重，该关键帧与pKF共同观测到的3d点数量
 */
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))  //std::map::count()函数只可能返回0或1
            mConnectedKeyFrameWeights[pKF]=weight;//若无pKF,则添加连接,并赋值
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;//若有pKF但值不同,则更新
        else
            return;
    }

    UpdateBestCovisibles();
}

/**
 * @brief 按照权重对连接的关键帧进行排序
 * 由于map结构没有sort函数，需要将元素取出放入一个pair组成的vector中，排序后放入
 * 
 * 更新后的变量存储在mvpOrderedConnectedKeyFrames和mvOrderedWeights中
 */
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));
    // 取出所有连接的关键帧及其权重，mConnectedKeyFrameWeights的类型为std::map<KeyFrame*,int>，
    // 按照(权重,关键帧)的形式存入vPairs(vPairs变量将共视的3D点数放在前面，利于排序)
    // 排完序之后,将结果依次存入lKFs和lWs,并存入mvpOrderedconnectedKeyFrames和mvOrderedWeights中
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs; //KeyFrames
    list<int> lWs;        //Weights
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }
    //权重从大到小排
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

/**
 * @brief 得到与该关键帧连接的关键帧
 * @return 连接的关键帧
 */
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

/**
 * @brief 得到与该关键帧连接的关键帧(已按权值排序)
 * @return 连接的关键帧
 */
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

/**
 * @brief 得到与该关键帧连接的前N个关键帧(已按权值排序)
 * 
 * 如果连接的关键帧少于N，则返回所有连接的关键帧
 * @param N 前N个
 * @return 连接的关键帧
 */
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

/**
 * @brief 得到与该关键帧连接的权重大于等于w的关键帧
 * @param w 权重
 * @return 连接的关键帧
 */rnegin
vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    // 在从小到大的排序数组里,upper_bound(begin,end,w)返回第一个大于w的,lower_bound(begin,end,num)返回第一个大于等于w的
    // 如果找到就返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin，得到找到数字在数组中的下标。
    // 如果是从大到小的数组中，则重载
    // lower_bound(begin,end,num,greater<type>()) 查找第一个小于等于num的数字
    // upper_bound(begin,end,num,greater<type>()) 查找第一个小于num的数字
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

/**
 * @brief 得到该关键帧与pKF的权重
 * @param  pKF 关键帧
 * @return     权重
 */
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

/**
 * @brief Add MapPoint to KeyFrame
 * @param pMP MapPoint
 * @param idx MapPoint在KeyFrame中的索引
 */
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

/**
 * @brief 关键帧中，大于等于minObs的MapPoints的数量
 * minObs就是一个阈值，大于minObs就表示该MapPoint是一个高质量的MapPoint
 * 一个高质量的MapPoint会被多个KeyFrame观测到，
 * @param  minObs 最小观测
 */
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)//minObs>0
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else//minObs<=0
                    nPoints++;
            }
        }
    }

    return nPoints;
}

/**
 * @brief Get MapPoint Matches
 *
 * 获取该关键帧的MapPoints
 */
vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

/**
 * @brief 更新图的连接
 * 
 * 1. 首先获得该关键帧的所有MapPoint点，统计观测到这些3d点的每个关键与其它所有关键帧之间的共视程度
 *    对每一个找到的关键帧，建立一条边，边的权重是该关键帧与当前关键帧公共3d点的个数。
 * 2. 并且该权重必须大于一个阈值，如果没有超过该阈值的权重，那么就只保留权重最大的边（与其它关键帧的共视程度比较高）
 * 3. 对这些连接按照权重从大到小进行排序，以方便将来的处理
 *    更新完covisibility图之后，如果没有初始化过，则初始化为连接权重最大的边（与其它关键帧共视程度最高的那个关键帧），类似于最大生成树
 */
void KeyFrame::UpdateConnections()
{
    // 在没有执行这个函数前，关键帧只和MapPoints之间有连接关系，这个函数可以更新关键帧之间的连接关系
    //===============1==================================
    map<KeyFrame*,int> KFcounter;// <关键帧,权重>，权重为其它关键帧与当前关键帧共视3d点的个数

    vector<MapPoint*> vpMP;

    {
        //获得该关键帧所有的3D点mvpMapPoints
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    // 通过3D点间接统计可以观测到这些3D点的所有关键帧之间的共视程度
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;
        
        // 对于每一个MapPoint点，observations记录了可以观测到该MapPoint的所有关键帧
        map<KeyFrame*,size_t> observations = pMP->GetObservations();
        //统计每一个关键帧都有多少关键帧与它存在共视关系，统计结果放在KFcounter
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId)//除去自身,自己与自己不算共视
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;
    //===============2==================================
    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    // vPairs记录与其它关键帧共视帧数大于th的关键帧
    // pair<int,KeyFrame*>将关键帧的权重写在前面，关键帧写在后面,方便后面排序
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;// 找到对应权重最大的关键帧（共视程度最高的关键帧）
            pKFmax=mit->first;
        }
        if(mit->second>=th)// 对应权重需要大于阈值，对这些关键帧建立连接
        {
            // 更新KFcounter中该关键帧的mConnectedKeyFrameWeights
            // 更新其它KeyFrame的mConnectedKeyFrameWeights，更新其它关键帧与当前帧的连接权重
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    // 如果没有超过阈值的权重，则对权重最大的关键帧建立连接
    if(vPairs.empty())
    {
        // 如果每个关键帧与它共视的关键帧的个数都少于th，
        // 那就只更新与其它关键帧共视程度最高的关键帧的mConnectedKeyFrameWeights
        // 这是对之前th这个阈值可能过高的一个补丁
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());// vPairs里存的都是相互共视程度比较高的关键帧和共视权重，由大到小
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

   //===============3==================================
    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        // 更新图的连接(权重)
        mConnectedKeyFrameWeights = KFcounter;//更新该KeyFrame的mConnectedKeyFrameWeights，更新当前帧与其它关键帧的连接权重
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        // 更新生成树的连接
        if(mbFirstConnection && mnId!=0)
        {
            // 初始化该关键帧的父关键帧为共视程度最高的那个关键帧
            mpParent = mvpOrderedConnectedKeyFrames.front();
            // 建立双向连接关系
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    // 这个地方是不是应该：(!mbToBeErased)，(wubo???)
    // SetBadFlag函数就是将mbToBeErased置为true，mbToBeErased就表示该KeyFrame被擦除了
    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)// mbNotErase表示不应该擦除该KeyFrame，于是把mbToBeErased置为true，表示已经擦除了，其实没有擦除
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);// 让其它的KeyFrame删除与自己的联系

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);// 让与自己有联系的MapPoint删除与自己的联系
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        //清空自己与其它关键帧之间的联系
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        // 如果这个关键帧有自己的孩子关键帧，告诉这些子关键帧，它们的父关键帧不行了，赶紧找新的父关键帧
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            // 遍历每一个子关键帧，让它们更新它们指向的父关键帧
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                // 子关键帧遍历每一个与它相连的关键帧（共视关键帧）
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                    // 如果该帧的子节点和父节点（祖孙节点）之间存在连接关系（共视）
                    // 举例：B-->A（B的父节点是A） C-->B（C的父节点是B） D--C（D与C相连） 
                    //      E--C（E与C相连） F--C（F与C相连） D-->A（D的父节点是A） E-->A（E的父节点是A）
                    //      现在B挂了，于是C在与自己相连的D、E、F节点中找到父节点指向A的D
                    //      此过程就是为了找到可以替换B的那个节点。
                    // 上面例子中，B为当前要设置为SetBadFlag的关键帧
                    //           A为spcit，也即sParentCandidates
                    //           C为pKF,pC，也即mspChildrens中的一个
                    //           D、E、F为vpConnected中的变量，由于C与D间的权重 比 C与E间的权重大，因此D为pP
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                // 因为父节点死了，并且子节点找到了新的父节点，子节点更新自己的父节点
                pC->ChangeParent(pP);
                // 因为子节点找到了新的父节点并更新了父节点，那么该子节点升级，作为其它子节点的备选父节点
                sParentCandidates.insert(pC);
                // 该子节点处理完毕
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        // 如果还有子节点没有找到新的父节点
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                // 直接把父节点的父节点作为自己的父节点
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const//r为边长
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // floor向下取整，mfGridElementWidthInv为每个像素占多少个格子
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));//X向最小的格子数
    if(nMinCellX>=mnGridCols)
        return vIndices;

    //ceil向上取整
    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));//X向最大的格子数
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            //对每个格子,取出其中的keypoint,判断是否在x和y轴方向上距(x,y)不超过r的范围内
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i 第i个keypoint
 * @return   3D点（相对于世界坐标系）
 */
cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        //  |u|=|fx*x/z+cx|   -->x=(u-cx)*z/fx
        //  |v|=|fy*y/z+cy|   -->y=(v-cy)*z/fy
        //  
        //  |X|     |x|
        //  |Y|=Rwc*|y|+twc
        //  |Z|     |z|
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        // 由相机坐标系转换到世界坐标系
        // Twc为相机坐标系到世界坐标系的变换矩阵
        // Twc.rosRange(0,3).colRange(0,3)取Twc矩阵的前3行与前3列
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

/**
 * @brief 评估当前关键帧场景深度，q=2表示中值
 * @param q q=2
 * @return Median Depth
 */
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;//世界坐标系下的点x3Dw在Tcw的作用下得到相机坐标系下的点的z坐标,即深度
            vDepths.push_back(z);
            //  |x| |R0 t0||X| |R0*X+t0|
            //  |y|=|R1 t1||Y|=|R1*Y+t1|
            //  |z| |R2 t2||Z| |R2*Z+t2|
            //  |1| |0   1||1| |    1  |
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];//若q为2,则返回深度的中值
}

} //namespace ORB_SLAM
